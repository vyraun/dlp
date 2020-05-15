import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as bp

class EncoderRNN(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embed_size, 
                 hidden_size,
                 dropout_rate,
                 num_layers,
                 bidirectional, embeddings=None):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size 
        self.input_size = vocab_size
        self.embed_size = embed_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(self.input_size, self.embed_size)
        if embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.dropout = nn.Dropout(self.dropout_rate)
        self.LSTM = nn.LSTM(self.embed_size, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout_rate, bidirectional=self.bidirectional)
        #for layer
        self.embedding_layerNorm = nn.LayerNorm(self.embed_size)
        self.hidden_layerNorm = nn.LayerNorm(self.hidden_size)
        self.transform_hidden = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, input, input_lengths):
        embedded = self.embedding(input)
        #embedded = self.dropout(embedded)
        #embedded = self.embedding_layerNorm(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        output, hidden = self.LSTM(packed, None)

        #hidden_list = list(hidden)
        #for i in range(len(hidden)):
        #    hidden_list[i] = self.hidden_layerNorm(hidden_list[i])
        #hidden = tuple(hidden_list)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        if self.bidirectional != True:
            #hidden = self.transform(hidden)
            return output, hidden
        else:
            hidden_final = torch.cat((hidden[0][0], hidden[0][1]), dim=1).unsqueeze(0)
            cell_final = torch.cat((hidden[1][0], hidden[1][1]), dim=1).unsqueeze(0)
            for i in range(1, self.num_layers):
                hidden_final = torch.cat((hidden_final, torch.cat((hidden[0][2*i], hidden[0][2*i + 1]), dim=1).unsqueeze(0)), dim=0)
                cell_final = torch.cat((cell_final, torch.cat((hidden[1][2*i], hidden[1][2*i + 1]), dim=1).unsqueeze(0)), dim=0)
        # Changes for Bidi to work
            cell_final = self.transform_hidden(cell_final)
            hidden_final = F.tanh(cell_final)
            return output, (hidden_final, cell_final)

class DecoderRNN(nn.Module):
    def __init__(self, 
                 embed_size, 
                 hidden_size, 
                 output_size,
                 dropout_rate,
                 num_layers,
                 attention_type='none',
                 bidirectional=True,
                 self_attention=False):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size 
        self.embed_size = embed_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.embedding = nn.Embedding(self.output_size, self.embed_size)
        self.embedding_layerNorm = nn.LayerNorm(self.embed_size)
        self.hidden_layerNorm = nn.LayerNorm(self.hidden_size)
        self.attention_type = attention_type
        self.bidirectional = bidirectional
        self.attention_type = attention_type
        if self.bidirectional == True:
            self.transform_hidden = nn.Linear(self.hidden_size * 2, self.hidden_size)
        else:
            self.transform_hidden = nn.Linear(self.hidden_size, self.hidden_size)

        # Calculate LSTM input size
        input_size = self.embed_size
        if self.attention_type != 'none':
          input_size += self.hidden_size

        if self.bidirectional == True:
            input_size += self.hidden_size

        if self_attention:
          input_size += self.hidden_size

        if self.attention_type == 'general':
                if self.bidirectional == True:
                    self.attention_layer = nn.Linear(2 * self.hidden_size, self.hidden_size)
                else:
                    self.attention_layer = nn.Linear(self.hidden_size, self.hidden_size)

        if self.attention_type == 'concat':
            self.attention_layer = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.v = nn.Linear(self.hidden_size, 1)

        self.LSTM = nn.LSTM(input_size, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout_rate)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        # Added for concatentation context vector with hidden, and passing it to softmax
        if self.bidirectional == True:
            self.output_linear = nn.Linear(self.hidden_size * 3, self.hidden_size)
        else:
            self.output_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.self_attention = self_attention

    def forward(self, encoder_outputs, hidden, output, prev_context, flag=0, output_lengths=None):
        embedded = self.embedding(output)
        #embedded = self.embedding_layerNorm(embedded)
        #embedded = F.relu(embedded)
        #embedded = self.dropout(embedded)
        rnn_input = embedded
        new_input = torch.cat((rnn_input, prev_context), dim=2)
        output, hidden = self.LSTM(new_input, hidden)

        if self.num_layers > 1:
            cur_hidden = hidden[0][-1:]
        else:
            cur_hidden = hidden[0]

        if self.attention_type == 'dot':
          # Multiply (B x 1 x H) * (B x H x S) = (B x 1 x S)
          cur_hidden = cur_hidden[-1:].transpose(0,1)
          encoder_hiddens = encoder_outputs.transpose(0,1).transpose(1,2)
          attn_weights = F.softmax(cur_hidden.bmm(encoder_hiddens), dim=2)

          # Determine encoder context, (B x 1 x S) * (B x S x H) = (B x 1 x H)
          encoder_contexts = attn_weights.bmm(encoder_outputs.transpose(0,1))

          # Concate with embedded
          rnn_input = torch.cat((rnn_input, encoder_contexts.transpose(0,1)), dim=2)

        elif self.attention_type == 'general':
          # Multiply (B x 1 x H) * (B x H x S) = (B x 1 x S)
          encoder_outputs2 = self.attention_layer(encoder_outputs)
          cur_hidden = cur_hidden[-1:].transpose(0,1)
          encoder_hiddens = encoder_outputs2.transpose(0,1).transpose(1,2)
          attn_weights = F.softmax(cur_hidden.bmm(encoder_hiddens), dim=2)

          # Determine encoder context, (B x 1 x S) * (B x S x H) = (B x 1 x H)
          encoder_contexts = attn_weights.bmm(encoder_outputs.transpose(0,1))

        elif self.attention_type == 'concat':
          # Multiply (B x 1 x H) * (B x H x S) = (B x 1 x S)
          attention_input  = torch.cat((cur_hidden.expand(encoder_outputs.size(0), -1, -1), encoder_outputs), dim=2)
          attention_outputs = self.v(F.tanh(self.attention_layer(attention_input)))
          attention_outputs = attention_outputs.transpose(0, 1).transpose(1, 2)
          attn_weights = F.softmax(attention_outputs, dim=2)

          # Determine encoder context, (B x 1 x S) * (B x S x H) = (B x 1 x H)
          encoder_contexts = attn_weights.bmm(encoder_outputs.transpose(0,1))

          # Concate with embedded

        """
        Self-attention implementation. Uncomment after hitting 27 BLEU, to avoid slowing down training.
        Expects decoder_outputs to be passed in.

        if self.self_attention:
          # Multiply (B x 1 x H) * (B x H x S) = (B x 1 x S)
          decoder_hiddens = decoder_outputs.transpose(0,1).transpose(1,2)
          dec_attn_weights = F.softmax(cur_hidden.bmm(decoder_hiddens), dim=2)

          # Determine decoder context, (B x 1 x S) * (B x S x H) = (B x 1 x H)
          decoder_contexts = dec_attn_weights.bmm(decoder_outputs.transpose(0,1))

          rnn_input = torch.cat((rnn_input, decoder_contexts))
        """

        #after the changes rnn_input is the same as embedded
        #output, hidden = self.LSTM(rnn_input, hidden)

        if self.num_layers > 1:
          cur_hidden = hidden[0][-1:]
        else:
          cur_hidden = hidden[0]
        output_final = F.tanh(self.output_linear(torch.cat((cur_hidden, encoder_contexts.transpose(0, 1)), dim=2)))
        output_final = self.dropout(output_final)
        output = self.out(output_final)
        #output = F.log_softmax(self.out(hidden[0]), dim=2)
        return output, hidden, encoder_contexts.transpose(0, 1)

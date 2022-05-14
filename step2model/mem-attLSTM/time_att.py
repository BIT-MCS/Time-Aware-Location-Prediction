from weeplaces.step2model.utils import *


class TimeAwareAtt(nn.Module):
    def __init__(self, dimensions, attention_type='general'):
        super(TimeAwareAtt, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, target_time, history_time, history_h):
        """
        Args:
            target_time (:class:`torch.FloatTensor` [batch size, dimensions]):
                Data by which to query the history_time.
            history_time (:class:`torch.FloatTensor` [batch size, history length, dimensions]):
                Data to be queried by the target_time.
            history_h (:class:`torch.FloatTensor` [batch size, history length, dimensions]):
                Data over which to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, dimensions]):
              Tensor containing the attended features.
        """
        if self.attention_type == "general":
            target_time = self.linear_in(target_time)

        # (batch_size, 1, dimensions) * (batch_size, dimensions, history_len) ->
        # (batch_size, 1, history_len)
        attention_scores = torch.bmm(target_time.unsqueeze(dim=1), history_time.transpose(1, 2).contiguous())

        # Compute weights across every history_time sequence
        attention_weights = self.softmax(attention_scores)

        # (batch_size, 1, history_len) * (batch size, history length, dimensions) ->
        # (batch_size, 1, dimensions)
        mix = torch.bmm(attention_weights, history_h)

        output = mix[:, 0]

        return output, attention_weights[:, 0]

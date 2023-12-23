import torch
from transformers import TransfoXLModel, TransfoXLConfig
class CustomTransformerXL(TransfoXLModel):
    def __init__(self, num_classes, mem_len, alpha, *args, **kwargs):
        # Initialize the configuration for Transformer-XL
        config = TransfoXLConfig()

        super().__init__(config)

        # Additional attributes specific to your application
        self.num_classes = num_classes
        self.mem_len = mem_len
        self.alpha = alpha

        self.classifier = torch.nn.Linear(config.d_model, num_classes)



    def forward(self, x, mem, is_last):
        batch_size, _, _ = x.size()
        x_reshaped = x.view(-1,batch_size)

        # 将转换后的 x 作为 input_ids
        input_ids = x_reshaped
        mems = mem.get('mems', None)

        outputs = super().forward(input_ids, mems=mems)

        if not is_last.all():
            mem['mems'] = outputs['mems']

        # Apply the classifier to the last hidden state
        logits = self.classifier(outputs.last_hidden_state) if self.classifier is not None else outputs.last_hidden_state

        return logits, mem


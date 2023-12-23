import torch
from transformers import TransfoXLModel, TransfoXLConfig
class CustomTransformerXL(TransfoXLModel):
    def __init__(self, num_classes, mem_len, alpha, *args, **kwargs):
        # Initialize the configuration for Transformer-XL
        config = TransfoXLConfig()
        # You may want to customize the config here based on args/kwargs
        # e.g., config.mem_len = mem_len

        # Initialize the model from the base class
        super().__init__(config)

        # Additional attributes specific to your application
        self.num_classes = num_classes
        self.mem_len = mem_len
        self.alpha = alpha

        # A classification head that takes the output of the transformer which is of size `d_model`
        # and outputs `num_classes` predictions
        self.classifier = torch.nn.Linear(config.d_model, num_classes)
        # Define a CNN as feature extractor (this is just a placeholder)
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            # Add more layers as required
        )
    # def forward(self, img, mem, is_last):
    #     input_ids = img  # Ensure img is appropriately preprocessed and tokenized
    #     mems = mem.get('mems', None)
    #
    #     outputs = super().forward(input_ids, mems=mems)
    #
    #     # Handle the logic for updating memory states and extracting the output
    #     if not is_last.all():
    #         mem['mems'] = outputs.mems
    #
    #     # If you added a classifier in the __init__, apply it to the last hidden state
    #     logits = self.classifier(
    #         outputs.last_hidden_state) if self.classifier is not None else outputs.last_hidden_state
    #
    #     return logits, mem
    def forward(self, img, mem, is_last):
        # Assume feature_extractor is a CNN that converts image to a token-like sequence
        features = self.feature_extractor(img)  # Convert image batch to sequence-like data
        input_ids = features.squeeze()  # Flatten the features if necessary

        # Ensure input_ids is a 2D tensor with size [sequence length, batch size]
        if input_ids.ndim == 3:
            input_ids = input_ids.view(input_ids.size(0), -1)  # [seq_len, batch_size*feature_size]

        mems = mem.get('mems', None)

        outputs = super().forward(input_ids, mems=mems)

        # Update the memory with the new mems if not the last segment
        if not is_last.all():
            mem['mems'] = outputs['mems']

        # Apply the classifier to the last hidden state
        logits = self.classifier(outputs.last_hidden_state) if self.classifier is not None else outputs.last_hidden_state

        return logits, mem

# # Usage:
# config = TransfoXLConfig()
# # Add any other configuration parameters required for your task
# model = CustomTransformerXL(config)

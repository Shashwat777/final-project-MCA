
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from torchvision import models
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, tokens, input_ids, input_mask, input_type_ids):
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example)

        if len(tokens_a) > seq_length - 2:
            tokens_a = tokens_a[0:(seq_length - 2)]

        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        features.append(
            InputFeatures(
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


def bert_feature(examples, model, tokenizer, seq_length=64):

    features = convert_examples_to_features(
            examples=examples, seq_length=seq_length, tokenizer=tokenizer)

    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(device)

    _, pooled_output  = model(input_ids, token_type_ids=None, attention_mask=input_mask)

    return pooled_output


class SMP_model(nn.Module):
    def __init__(self):
        super(SMP_model, self).__init__()

        self.img_feature = nn.Sequential(*list(models.resnet101(pretrained=True).children())[:-1])

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

        self.conv = nn.Conv2d(3,1,1)
        self.conv.weight.data.normal_(1/3,0.01)

        self.img_classifier = nn.Sequential(
            nn.Linear(2048,128),
            )
        self.text_classifier = nn.Sequential(
            nn.Linear(768,128),
            )
        self.meta_classifier = nn.Sequential(
            nn.Linear(10,128),
            )                               
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(384,affine=False),
            nn.ReLU(),
            nn.Linear(384,256),
            nn.BatchNorm1d(256,affine=False),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.BatchNorm1d(128,affine=False),
            nn.ReLU(),
            nn.Linear(128,1),
            )


    def forward(self, img, texts, meta):

        meta_features = self.meta_classifier(meta)

        with torch.no_grad():
            img_features = self.img_feature(img).squeeze()
        img_features = self.img_classifier(img_features)

        text_features_list = []
        for text in texts:
            with torch.no_grad():
                text_features = bert_feature(text, self.bert_model, self.tokenizer)
                text_features_list.append(self.text_classifier(text_features))
        text_features = torch.stack(text_features_list,1).unsqueeze(3)
        text_features = self.conv(text_features).permute(0,2,1,3).squeeze()     
        
        x = torch.cat([img_features,text_features,meta_features],1)
        out = self.classifier(x)
        return out

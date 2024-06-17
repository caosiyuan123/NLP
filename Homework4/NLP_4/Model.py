# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cpu")

class S2SArgs:
    #模型参数
    embedding_dim = 256
    hidden_dim = 512
    num_epochs = 10
    batch_size = 4
    learning_rate = 0.01
    save_path = '.\\models\\S2SModel.pth'

class TfArgs:
    # 模型参数
    embedding_dim  = 256
    nhead = 8
    num_encoder_layers = 4
    num_decoder_layers = 4
    dim_feedforward = 1024
    # 训练循环
    num_epochs = 20
    batch_size = 4
    learning_rate = 0.001
    save_path = '.\\models\\TfModel.pth'

class S2SModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(S2SModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src, tgt):
        # src的形状：(batch_size,num_steps,embed_size)
        src_embedded = self.embedding(src)
        # tgt的形状：(batch_size,num_steps,embed_size)
        tgt_embedded = self.embedding(tgt)
        _, (hidden, cell) = self.encoder(src_embedded)
        output, _ = self.decoder(tgt_embedded, (hidden, cell))
        return self.fc_out(output)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim , nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim )
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embedding_dim ))
        self.transformer = nn.Transformer(embedding_dim , nhead, num_encoder_layers, num_decoder_layers,
                                          dim_feedforward, batch_first=True)
        self.fc_out = nn.Linear(embedding_dim , vocab_size)
        self.max_seq_length = max_seq_length

    def forward(self, src, tgt):
        src = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        memory = self.transformer.encoder(src)
        output = self.transformer.decoder(tgt, memory)
        return self.fc_out(output)

def S2SModel_train(trainDataset, dictionary, test_Len):
    # 创建模型
    vocab_size = len(dictionary)
    model = S2SModel(vocab_size=vocab_size,
                         embedding_dim=S2SArgs.embedding_dim,
                         hidden_dim=S2SArgs.hidden_dim).to(device)

    # 创建数据集和数据加载器
    dataloader = DataLoader(trainDataset, batch_size=S2SArgs.batch_size, shuffle=True, drop_last=True)

    # 训练Seq2Seq
    criterion = nn.CrossEntropyLoss()  # pad token id
    optimizer = optim.Adam(model.parameters(), lr=S2SArgs.learning_rate)
    for epoch in range(S2SArgs.num_epochs):
        model.train()
        for sentence_ids in dataloader:
            optimizer.zero_grad()
            sentence_ids = sentence_ids.to(device)
            src = sentence_ids[:, :test_Len]
            tgt_input = sentence_ids[:, test_Len:-1]
            tgt_output = sentence_ids[:, test_Len+1:]
            outputs = model(src, tgt_input)

            # 计算损失
            targets = torch.zeros_like(outputs, device=device)
            for i in range(targets.size(0)):
                for j in range(targets.size(1)):
                    if tgt_output[i][j] != 2:  # padding id
                        targets[i, j, tgt_output[i][j]] = 1
            loss = criterion(outputs, targets).to(device)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch + 1}/{S2SArgs.num_epochs}, Loss: {loss.item()}')
    print("Training complete!")
    torch.save(model, S2SArgs.save_path)

def TfModel_train(train_dataset, dictionary, test_len, max_len):
    # 创建模型
    vocab_size = len(dictionary)
    model = TransformerModel(vocab_size=vocab_size,
                             embedding_dim=TfArgs.embedding_dim ,
                             nhead=TfArgs.nhead,
                             num_encoder_layers=TfArgs.num_encoder_layers,
                             num_decoder_layers=TfArgs.num_decoder_layers,
                             dim_feedforward=TfArgs.dim_feedforward,
                             max_seq_length=max_len).to(device)

    # 创建数据集和数据加载器
    dataloader = DataLoader(train_dataset, batch_size=TfArgs.batch_size, shuffle=True, drop_last=True)

    # 训练Transformer，创建loss和优化器
    criterion = nn.CrossEntropyLoss()  # pad token id
    optimizer = optim.Adam(model.parameters(), lr=TfArgs.learning_rate )
    for epoch in range(TfArgs.num_epochs):
        model.train()
        for sentence_ids in dataloader:
            optimizer.zero_grad()
            sentence_ids = sentence_ids.to(device)
            src = sentence_ids[:, :test_len]
            tgt_input = sentence_ids[:, test_len:-1]
            tgt_output = sentence_ids[:, test_len+1:]

            outputs = model(src, tgt_input)

            #计算损失
            targets = torch.zeros_like(outputs, device=device)
            for i in range(targets.size(0)):
                for j in range(targets.size(1)):
                    if tgt_output[i][j] != 2:  # padding id
                        targets[i, j, tgt_output[i][j]] = 1
            loss = criterion(outputs, targets).to(device)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch + 1}/{TfArgs.num_epochs}, Loss: {loss.item()}')
    print("Training complete!")
    torch.save(model, TfArgs.save_path)

def S2SModel_test(test_dataset, dictionary, test_len, max_len):
    model = torch.load(S2SArgs.save_path).to(device)
    model.eval()
    dataloader = DataLoader(test_dataset, batch_size=1)

    for ground_ids in dataloader:
        test_ids = ground_ids[:, :test_len].to(device)
        test_ids = test_ids.to(device)
        generate_text_ids = test_ids
        for _ in range(max_len):
            with torch.no_grad():
                outputs = model(test_ids, generate_text_ids)
                next_token_id = torch.argmax(outputs[:, -1, :], dim=-1).unsqueeze(0)
                test_ids = torch.cat((test_ids, next_token_id), dim=-1)
                generate_text_ids = torch.cat((generate_text_ids, next_token_id), dim=-1)

            if next_token_id.squeeze(0) == dictionary.token2id['<eos>']:
                break
        input_text = [dictionary[textId.item()] for textId in test_ids.squeeze_(0)[:test_len]]
        generate_text = [dictionary[textId.item()] for textId in generate_text_ids.squeeze_(0)]
        print('InputData: ', input_text)
        print('OutputData: ', generate_text)

def TfModel_test(testDataset, dictionary, test_len, max_len):
    model = torch.load(TfArgs.save_path).to(device)
    model.eval()
    dataloader = DataLoader(testDataset, batch_size=1)

    for ground_ids in dataloader:
        test_ids = ground_ids[:, :test_len].to(device)
        test_ids = test_ids.to(device)
        generate_text_ids = test_ids
        for _ in range(max_len - test_len):
            with torch.no_grad():
                outputs = model(test_ids, generate_text_ids)
                next_token_id = torch.argmax(outputs[:, -1, :], dim=-1).unsqueeze(0)
                test_ids = torch.cat((test_ids, next_token_id), dim=-1)
                generate_text_ids = torch.cat((generate_text_ids, next_token_id), dim=-1)

            if next_token_id.squeeze(0) == dictionary.token2id['<eos>']:
                break
        input_text = [dictionary[textId.item()] for textId in test_ids.squeeze_(0)[:test_len]]
        generate_text = [dictionary[textId.item()] for textId in generate_text_ids.squeeze_(0)]
        print('InputData: ', input_text)
        print('OutputData: ', generate_text)
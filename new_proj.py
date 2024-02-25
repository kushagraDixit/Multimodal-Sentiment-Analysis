import pandas as pd
import av
import numpy as np
from transformers import AutoImageProcessor, VideoMAEImageProcessor, VideoMAEForVideoClassification, VideoMAEModel
import torch
from transformers import AutoTokenizer, RobertaModel,RobertaForSequenceClassification
from tqdm import tqdm
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import argparse


sentiment_encoded = {'neutral': 0, 'positive': 1, 'negative': 2}
emotion_encoded = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}

'''class MultimodalClassifier(torch.nn.Module):
    def __init__(self, num_frames, num_labels, droput_prob):
        super().__init__()
        
        self.mod_video = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base", num_frames=num_frames, hidden_dropout_prob=droput_prob, attention_probs_dropout_prob=droput_prob)
        self.mod_text = RobertaModel.from_pretrained("roberta-base", hidden_dropout_prob=droput_prob, attention_probs_dropout_prob=droput_prob)
        self.con_vid = torch.nn.Linear(784 * 768, 768, bias=False)
        self.classifier = torch.nn.Linear(768*2, num_labels, bias=True)
        self.drop = torch.nn.Dropout(droput_prob)
        self.drop_vid = torch.nn.Dropout(droput_prob)

    def forward(self, inp_px, inp_ids, am):
        out_video = self.mod_video(pixel_values=inp_px)
        out_video = out_video.last_hidden_state.view(out_video.last_hidden_state.size()[0], -1)
        x1 = self.con_vid(out_video)
        x1 = self.drop_vid(x1)

        out_text = self.mod_text(input_ids=inp_ids, attention_mask=am)
        x2 = out_text.pooler_output

        combined = torch.cat((x1, x2), 1)

        drop_out = self.drop(combined)

        output = self.classifier(drop_out)

        return output'''

class MultimodalClassifier(torch.nn.Module):
    def __init__(self, num_frames, num_labels, droput_prob, sent_encoded, id2label):
        super().__init__()
        
        self.mod_video = VideoMAEForVideoClassification.from_pretrained(
                            "MCG-NJU/videomae-base",
                            num_frames=num_frames,
                            hidden_dropout_prob=droput_prob,
                            attention_probs_dropout_prob=droput_prob,
                            label2id=sent_encoded,
                            id2label=id2label,
                            ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
                    )
        self.mod_text = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels, hidden_dropout_prob=droput_prob, attention_probs_dropout_prob=droput_prob)
        
        self.classifier = torch.nn.Linear(num_labels*2, num_labels, bias=True)

    def forward(self, inp_px, inp_ids, am, ls):
        
        out_text = self.mod_text(input_ids=inp_ids, attention_mask=am, labels=ls)

        out_video = self.mod_video(pixel_values=inp_px,labels=ls)

        x1 = out_text.logits

        x2 = out_video.logits

        combined = torch.cat((x1, x2), 1)

        output = self.classifier(combined)

        return output

def getDataFrame(df_csv, path, start, end):
    df = pd.DataFrame(columns=['Sr No.', 'Utterance', 'Emotion', 'Sentiment', 'VideoPath'])
    
    if(end>df_csv.index.stop):
        end = df_csv.index.stop

    for i in range (start,end):
        name = 'dia' + str(df_csv['Dialogue_ID'][i]) + '_utt' + str(df_csv['Utterance_ID'][i]) + '.mp4'
        df.loc[len(df.index)] = [df_csv['Sr No.'][i], df_csv['Utterance'][i], df_csv['Emotion'][i], df_csv['Sentiment'][i], path+name]

    df['Sentiment'] = df.Sentiment.map(sentiment_encoded)
    df['Emotion'] = df.Emotion.map(emotion_encoded)

    return df

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            cnt = np.count_nonzero(indices == i)
            for k in range(cnt):
                frames.append(frame)
    
    ret_frames = np.stack([x.to_ndarray(format="rgb24") for x in frames])    

    return ret_frames


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    if(converted_len<seg_len):
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    else:
        start_idx = 0
        end_idx = seg_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def getDataLoader(df, image_processor, tokenizer, clip_length, batch_size):
    videos = []
    label_sentiment = []
    label_emotions = []
    tokens = []

    for i in tqdm(range(len(df.index))):
        try:
            file_path = df['VideoPath'][i]
            #print(file_path)
            container = av.open(file_path)
            indices = sample_frame_indices(clip_len=clip_length, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
            #print(f"for {i} : {indices}")
            video = read_video_pyav(container, indices)
            videos.append(list(video))
            label_sentiment.append(df['Sentiment'][i])
            label_emotions.append(df['Emotion'][i])
            tokens.append(df['Utterance'][i])
        except:
            print(f"error with video : {df['VideoPath'][i]}")

    label_emotions = np.asarray(label_emotions)
    label_sentiment = np.asarray(label_sentiment)

    input_videos = image_processor(videos, return_tensors="pt",padding="max_length")
    label_sentiment = torch.Tensor(label_sentiment).type(torch.LongTensor)
    label_emotions = torch.Tensor(label_emotions).type(torch.LongTensor)

    input_text = tokenizer(tokens, return_tensors="pt", truncation=True, padding=True)
    input_ids = torch.Tensor(input_text['input_ids']).type(torch.LongTensor)
    attention_mask = torch.Tensor(input_text['attention_mask'])

    dataset = torch.utils.data.TensorDataset(input_videos['pixel_values'],input_ids,attention_mask,label_sentiment,label_emotions)
    loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    return loader

def trainModelVideo(model_video, optimizer_video, loss_fn, train_loader, dev_loader, max_epochs, device):
    best_perf_dict = {"dev_accuracy": 0, "dev_loss": sys.maxsize, "epoch":0}

    for ep in range(1, max_epochs + 1):
        print(f"Epoch {ep}")
        train_loss = []  

        # Training Loop
        for inp_px, inp_id, am, ls, le in tqdm(train_loader):
            model_video.train()
            optimizer_video.zero_grad()
            out = model_video(pixel_values=inp_px.to(device),labels=ls.to(device))
            loss = loss_fn(out.logits, ls.to(device))
            loss.backward()
            optimizer_video.step()
            train_loss.append(loss.cpu().item())

        print(f"Average training batch loss Video Model: {np.mean(train_loss)}")

        # Evaluation Loop
        gold_labels = []
        pred_labels = []
        dev_loss = []
        for inp_px, inp_id, am, ls, le in tqdm(dev_loader):
            model_video.eval()

            with torch.no_grad():
                out = model_video(pixel_values=inp_px.to(device),labels=ls.to(device))
                logits = out.logits
                preds = logits.argmax(-1)
                loss = out.loss
                pred_labels.extend(preds.cpu().tolist())
                gold_labels.extend(ls.tolist())
                dev_loss.append(loss.cpu().item())

        dev_loss_value = np.mean(dev_loss)
        print(f"Average dev batch loss Video Model: {dev_loss_value}")
        dev_accuracy =  accuracy_score(gold_labels,pred_labels)
        print(f"Dev Accuracy Video Model: {dev_accuracy}")
        if(dev_accuracy > best_perf_dict["dev_accuracy"]):
            best_perf_dict["dev_accuracy"] = dev_accuracy
            best_perf_dict["dev_loss"] = dev_loss_value
            best_perf_dict["epoch"] = ep
            torch.save({
                "model_param": model_video.state_dict(),
                "optim_param": optimizer_video.state_dict(),
                "dev_loss": dev_loss_value,
                "epoch": ep
            }, f"./Models/model_video_{ep}")
            best_path = f"./Models/model_video_{ep}"

    print(f"""\nBest Dev Accuracy of {best_perf_dict["dev_accuracy"]} at epoch {best_perf_dict["epoch"]} with Dev Loss of {best_perf_dict["dev_loss"]}""")
    return best_path

def trainModelText(model_text, optimizer_text, loss_fn, train_loader, dev_loader, max_epochs, device):
    best_perf_dict = {"dev_accuracy": 0, "dev_loss": sys.maxsize, "epoch":0}

    for ep in range(1, max_epochs + 1):
        print(f"Epoch {ep}")
        train_loss = []  

        # Training Loop
        for inp_px, inp_id, am, ls, le in tqdm(train_loader):
            model_text.train()
            optimizer_text.zero_grad()
            out = model_text(input_ids=inp_id.to(device), attention_mask=am.to(device), labels=ls.to(device))
            loss = loss_fn(out.logits, ls.to(device))
            loss.backward()
            optimizer_text.step()
            train_loss.append(loss.cpu().item())

        print(f"Average training batch loss Text Model: {np.mean(train_loss)}")

        # Evaluation Loop
        gold_labels = []
        pred_labels = []
        dev_loss = []
        for inp_px, inp_id, am, ls, le in tqdm(dev_loader):
            model_text.eval()

            with torch.no_grad():
                out = model_text(input_ids=inp_id.to(device), attention_mask=am.to(device), labels=ls.to(device))
                logits = out.logits
                preds = logits.argmax(-1)
                loss = out.loss
                pred_labels.extend(preds.cpu().tolist())
                gold_labels.extend(ls.tolist())
                dev_loss.append(loss.cpu().item())

        dev_loss_value = np.mean(dev_loss)
        print(f"Average dev batch loss Text Model: {dev_loss_value}")
        dev_accuracy =  accuracy_score(gold_labels,pred_labels)
        print(f"Dev Accuracy Text Model: {dev_accuracy}")
        if(dev_accuracy > best_perf_dict["dev_accuracy"]):
            best_perf_dict["dev_accuracy"] = dev_accuracy
            best_perf_dict["dev_loss"] = dev_loss_value
            best_perf_dict["epoch"] = ep
            torch.save({
                "model_param": model_text.state_dict(),
                "optim_param": optimizer_text.state_dict(),
                "dev_loss": dev_loss_value,
                "epoch": ep
            }, f"./Models/model_text_{ep}")
            best_path = f"./Models/model_text_{ep}"

    print(f"""\nBest Dev Accuracy of {best_perf_dict["dev_accuracy"]} at epoch {best_perf_dict["epoch"]} with Dev Loss of {best_perf_dict["dev_loss"]}""")
    return best_path

def trainModelMultimodal(model_multimodal, optimizer_mm, loss_fn, train_loader, dev_loader, max_epochs, device):
    best_perf_dict = {"dev_accuracy": 0, "dev_loss": sys.maxsize, "epoch":0}

    for ep in range(1, max_epochs + 1):
        print(f"Epoch {ep}")
        train_loss = []  

        # Training Loop
        for inp_px, inp_id, am, ls, le in tqdm(train_loader):
            model_multimodal.train()
            optimizer_mm.zero_grad()
            # out = model_multimodal(inp_px.to(device), inp_id.to(device), am.to(device))
            out = model_multimodal(inp_px.to(device), inp_id.to(device), am.to(device), ls.to(device))
            loss = loss_fn(out, ls.to(device))
            loss.backward()
            optimizer_mm.step()
            train_loss.append(loss.cpu().item())

        print(f"Average training batch loss Multimodal: {np.mean(train_loss)}")

        # Evaluation Loop
        gold_labels = []
        pred_labels = []
        dev_loss = []
        for inp_px, inp_id, am, ls, le in tqdm(dev_loader):
            model_multimodal.eval()

            with torch.no_grad():
                #out = model_multimodal(inp_px.to(device), inp_id.to(device), am.to(device))
                out = model_multimodal(inp_px.to(device), inp_id.to(device), am.to(device), ls.to(device))
                preds = out.argmax(-1)
                loss = loss_fn(out, ls.to(device))
                pred_labels.extend(preds.cpu().tolist())
                gold_labels.extend(ls.tolist())
                dev_loss.append(loss.cpu().item())


        dev_loss_value = np.mean(dev_loss)
        print(f"Average dev batch loss Multimodal: {dev_loss_value}")
        dev_accuracy =  accuracy_score(gold_labels,pred_labels)
        print(f"Dev Accuracy Multimodal Model: {dev_accuracy}")
        if(dev_accuracy > best_perf_dict["dev_accuracy"]):
            best_perf_dict["dev_accuracy"] = dev_accuracy
            best_perf_dict["dev_loss"] = dev_loss_value
            best_perf_dict["epoch"] = ep
            torch.save({
                "model_param": model_multimodal.state_dict(),
                "optim_param": optimizer_mm.state_dict(),
                "dev_loss": dev_loss_value,
                "epoch": ep
            }, f"./Models/model_mm_{ep}")
            best_path = f"./Models/model_mm_{ep}"

    print(f"""\nBest Dev Accuracy of {best_perf_dict["dev_accuracy"]} at epoch {best_perf_dict["epoch"]} with Dev Loss of {best_perf_dict["dev_loss"]}""")
    return best_path

def checkTestAccuracyVideo(model_path,model_video,test_loader,device):

    checkpoint = torch.load(model_path)
    model_video.load_state_dict(checkpoint["model_param"])

    gold_labels = []
    pred_labels = []
    for inp_px, inp_id, am, ls, le in tqdm(test_loader):
        model_video.eval()
        with torch.no_grad():
            out = model_video(pixel_values=inp_px.to(device),labels=ls.to(device))
            logits = out.logits
            preds = logits.argmax(-1)
            pred_labels.extend(preds.cpu().tolist())
            gold_labels.extend(ls.tolist())

    test_accuracy =  accuracy_score(gold_labels,pred_labels)
    print(f"Test Accuracy Video Model: {test_accuracy}")
    test_f1 = f1_score(gold_labels, pred_labels, average='macro')
    print(f"Test F1 Score: {test_f1}\n")

def checkTestAccuracyText(model_path,model_text,test_loader,device):

    checkpoint = torch.load(model_path)
    model_text.load_state_dict(checkpoint["model_param"])

    gold_labels = []
    pred_labels = []
    for inp_px, inp_id, am, ls, le in tqdm(test_loader):
        model_text.eval()
        with torch.no_grad():
            out = model_text(input_ids=inp_id.to(device), attention_mask=am.to(device), labels=ls.to(device))
            logits = out.logits
            preds = logits.argmax(-1)
            pred_labels.extend(preds.cpu().tolist())
            gold_labels.extend(ls.tolist())

    test_accuracy =  accuracy_score(gold_labels,pred_labels)
    print(f"Test Accuracy Text Model: {test_accuracy}")

def checkTestAccuracyMultimodal(model_path,model_multimodal,test_loader,device):

    checkpoint = torch.load(model_path)
    model_multimodal.load_state_dict(checkpoint["model_param"])

    gold_labels = []
    pred_labels = []
    for inp_px, inp_id, am, ls, le in tqdm(test_loader):
        model_multimodal.eval()
        with torch.no_grad():
            
            #out = model_multimodal(inp_px.to(device), inp_id.to(device), am.to(device))
            out = model_multimodal(inp_px.to(device), inp_id.to(device), am.to(device), ls.to(device))
            preds = out.argmax(-1)
            pred_labels.extend(preds.cpu().tolist())
            gold_labels.extend(ls.tolist())

    test_accuracy =  accuracy_score(gold_labels,pred_labels)
    print(f"Test Accuracy Multimodal Model: {test_accuracy}")

def LoadDataLoaders():
    path = '/scratch/general/vast/u1472614/byop_data_16/'

    train1_path = path + 'train_loader_1.pt'
    train2_path = path + 'train_loader_2.pt'
    train3_path = path + 'train_loader_3.pt'
    train4_path = path + 'train_loader_4.pt'

    train1 = torch.load(train1_path)
    train2 = torch.load(train2_path)
    train3 = torch.load(train3_path)
    train4 = torch.load(train4_path)

    train_loader = [d for dl in [train1, train2, train3, train4] for d in dl]

    dev1_path = path + 'dev_loader_1.pt'
    dev2_path = path + 'dev_loader_2.pt'
    dev3_path = path + 'dev_loader_3.pt'
    dev4_path = path + 'dev_loader_4.pt'

    dev1 = torch.load(dev1_path)
    dev2 = torch.load(dev2_path)
    dev3 = torch.load(dev3_path)
    dev4 = torch.load(dev4_path)

    dev_loader = [d for dl in [dev1, dev2, dev3, dev4] for d in dl]

    test1_path = path + 'test_loader_1.pt'
    test2_path = path + 'test_loader_2.pt'
    test3_path = path + 'test_loader_3.pt'
    test4_path = path + 'test_loader_4.pt'

    test1 = torch.load(test1_path)
    test2 = torch.load(test2_path)
    test3 = torch.load(test3_path)
    test4 = torch.load(test4_path)

    test_loader = [d for dl in [test1, test2, test3, test4] for d in dl]

    return train_loader, dev_loader, test_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='The Learning Rate of the Model')
    parser.add_argument('--batch_size', default=16, type=int, help='The batch size for training')
    parser.add_argument('--num_frames', default=8, type=int, help='Num of frames in the data preprocessing step for visual model')
    parser.add_argument('--epochs', default=10, type=int, help='Total number of epochs for training')
    parser.add_argument('--seed', default=64, type=int, help='Initial seed')
    parser.add_argument('--iteration', default=1, type=int, help='Iteration for creating data loader')
    parser.add_argument('--dropout', default=0.3, type=float, help='Dropout value for al the models')
    parser.add_argument('--model', default='multimodal', type=str, help='Which model to run visual/text/multimodal/dataloader')

    args = parser.parse_args()
    LEARNING_RATE , MAX_EPOCHS, BATCH_SIZE , NUM_FRAMES , SEED, ITERATION, DROPOUT, MODEL = (args.learning_rate , args.epochs,
                                                                    args.batch_size , args.num_frames, 
                                                                    args.seed, args.iteration, args.dropout, args.model)
    
    np.random.seed(0)
    torch.manual_seed(SEED)
    model_ckpt = "MCG-NJU/videomae-base"

    if MODEL=='dataloader':
        print("Reading CSVs")

        df_mma_train = pd.read_csv('/uufs/chpc.utah.edu/common/home/u1472614/WORK/NLP-with-Deep-Learning/DATA/meld/train_sent_emo.csv')
        df_mma_dev = pd.read_csv('/uufs/chpc.utah.edu/common/home/u1472614/WORK/NLP-with-Deep-Learning/DATA/meld/dev_sent_emo.csv')
        df_mma_test = pd.read_csv('/uufs/chpc.utah.edu/common/home/u1472614/WORK/NLP-with-Deep-Learning/DATA/meld/test_sent_emo.csv')

        print(f"Total Train rows : {df_mma_train.index.stop}")
        print(f"Total Dev rows : {df_mma_dev.index.stop}")
        print(f"Total Test rows : {df_mma_test.index.stop}")

        path_train = '/uufs/chpc.utah.edu/common/home/u1472614/WORK/NLP-with-Deep-Learning/DATA/meld/train_splits/'
        path_dev = '/uufs/chpc.utah.edu/common/home/u1472614/WORK/NLP-with-Deep-Learning/DATA/meld/dev_splits_complete/'
        path_test = '/uufs/chpc.utah.edu/common/home/u1472614/WORK/NLP-with-Deep-Learning/DATA/meld/output_repeated_splits_test/'

        print("getting dataframes....")

        train_start = (ITERATION-1)*2500
        train_end = ITERATION*2500

        dev_start = (ITERATION-1)*280
        dev_end = ITERATION*280

        test_start = (ITERATION-1)*655
        test_end = ITERATION*655

        df_train = getDataFrame(df_mma_train, path_train,train_start,train_end)
        df_dev = getDataFrame(df_mma_dev, path_dev, dev_start, dev_end)
        df_test = getDataFrame(df_mma_test, path_test, test_start, test_end)

        image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        print("Creating Train Loader")
        train_loader = getDataLoader(df_train, image_processor=image_processor, tokenizer=tokenizer, clip_length=NUM_FRAMES, batch_size=BATCH_SIZE)
        torch.save(train_loader, f'./dataLoader/train_loader_{ITERATION}.pt')
        
        print("Creating Dev Loader")
        dev_loader = getDataLoader(df_dev, image_processor=image_processor, tokenizer=tokenizer, clip_length=NUM_FRAMES, batch_size=BATCH_SIZE)
        torch.save(dev_loader, './dataLoader/dev_loader_{ITERATION}.pt')

        print("Creating Test Loader")  
        test_loader = getDataLoader(df_test, image_processor=image_processor, tokenizer=tokenizer, clip_length=NUM_FRAMES, batch_size=BATCH_SIZE)
        torch.save(test_loader, './dataLoader/test_loader_{ITERATION}.pt')


    id2label = {i: label for label, i in sentiment_encoded.items()}
    print(f"Unique classes: {list(sentiment_encoded.keys())}.")

    print("Loading Models.....")
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = torch.nn.CrossEntropyLoss()

    if MODEL=='visual':
        train_loader, dev_loader, test_loader = LoadDataLoaders()
        
        model_video = VideoMAEForVideoClassification.from_pretrained(
        model_ckpt,
        num_frames=NUM_FRAMES,
        hidden_dropout_prob=DROPOUT,
        attention_probs_dropout_prob=DROPOUT,
        label2id=sentiment_encoded,
        id2label=id2label,
        ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        ).to(device)

        optimizer_video = torch.optim.Adam(model_video.parameters(), LEARNING_RATE)

        best_video_model_path = trainModelVideo(model_video, optimizer_video, loss_fn, train_loader, dev_loader, MAX_EPOCHS, device)

        checkTestAccuracyVideo(best_video_model_path, model_video, test_loader, device)
    
    if MODEL=='text':
        train_loader, dev_loader, test_loader = LoadDataLoaders()

        model_text = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3, hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.3).to(device)
        
        optimizer_text = torch.optim.Adam(model_text.parameters(), LEARNING_RATE)
    
        best_text_model_path = trainModelText(model_text, optimizer_text, loss_fn, train_loader, dev_loader, MAX_EPOCHS, device)

        checkTestAccuracyText(best_text_model_path, model_text, test_loader, device)

    if MODEL=='multimodal':
        train_loader, dev_loader, test_loader = LoadDataLoaders()

        model_multimodal = MultimodalClassifier(num_frames=NUM_FRAMES, num_labels=3, droput_prob=DROPOUT, sent_encoded=sentiment_encoded, id2label=id2label).to(device)
        
        optimizer_mm = torch.optim.Adam(model_multimodal.parameters(), LEARNING_RATE)
    
        best_mm_model_path = trainModelMultimodal(model_multimodal, optimizer_mm, loss_fn, train_loader, dev_loader, MAX_EPOCHS, device)

        checkTestAccuracyMultimodal(best_mm_model_path, model_multimodal, test_loader, device)
    

    

    

    

    

    
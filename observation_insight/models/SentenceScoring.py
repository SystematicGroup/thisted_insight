import numpy as np
from loguru import logger
from scipy.special import softmax
import pickle
from sklearn.metrics import accuracy_score
from observation_insight.data.DataTransformer import DatasetDict
import observation_insight.app.core.config as cfg

from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
import os
os.environ["WANDB_DISABLED"] = "true" # for not signing in to anything weird


class SentenceScoring():
    def __init__(self):
        """
        Constructs all necessary attributes for the sentence scoring object.

        Input
        ---------
            checkpoint: str
                Model for tokenization and fine-tuning
        
        Returns
        ---------
            model: Pretrained model from Huggingface
        """
        self.model = None

    def compute_metrics(self, prediction):    
        logits, labels = prediction
        pred = np.argmax(logits, axis=1)
        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred)
        precision = precision_score(y_true=labels, y_pred=pred)
        f1 = f1_score(y_true=labels, y_pred=pred)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, 'logits': logits} 

    def _freeze_layers(self, last_no_layers:int):
        """
        Freezing all but the last x layers

        Input
        ---------
            last_no_layers: int
                The number of layers that should not be frozen.
        """
        all_layers = []
        for name, param in self.model.named_parameters():
            if param.requires_grad == True: # layers that will be fine-tuned
                all_layers.append(name)

        unfreeze_layers = all_layers[-last_no_layers:]

        # Freeze all layers if not in unfreeze_layers list
        for name, param in self.model.named_parameters():
            if name not in unfreeze_layers:
                param.requires_grad = False

        print(">> Layers that are unfrozen:")
        for name, param in self.model.named_parameters():
            if param.requires_grad == True: # layers that will be fine-tuned
                print(name)

    def fine_tuning(self, train_dataset, val_dataset, train_args=None, freeze_layers=True, last_no_layers=None):
        """
        Fine-tuning pretrained (BERT) model with the Trainer API

        Input
        ---------
            train_dataset and val_dataset: datasets.arrow_dataset.Dataset
                Tokenized dataset with "features" and "num_rows"
            train_args: TrainingArguments()
                Arguments used for training, e.g. learning rate, batch size, no. of epochs

        Returns
        ---------
            trainer: transformers.trainer.Trainer
                Fine-tuned version of a pretrained model
        """
        if train_args == None:
            training_args = TrainingArguments("test_trainer")
        else:
            training_args = train_args

        if freeze_layers:
            self._freeze_layers(last_no_layers)

        trainer = Trainer(
            model = self.model, # pretrained model
            args = training_args, 
            train_dataset = train_dataset, 
            eval_dataset = val_dataset,
            compute_metrics = self.compute_metrics,
        )
        trainer.train()

        return trainer
    
    def evaluate_model(self, trainer, save=True, dir=None):
        eval = trainer.evaluate()
        if save:
            with open(f'{dir}/evaluation.pkl', 'wb') as f:
                pickle.dump(eval, f)
        return eval
    
    def predict(self, trainer, dataset, save=True, dir=None):
        pred = trainer.predict(test_dataset=dataset)
        if save:
            with open(f'{dir}/prediction.pkl', 'wb') as f:
                pickle.dump(pred.metrics, f)
        return pred

    def load_model(self, dir:str, num_labels=2):
        model = AutoModelForSequenceClassification.from_pretrained(dir, num_labels=num_labels)
        test_args = TrainingArguments(
            output_dir=cfg.LOGFILE_DIR,
            do_train = False,
            do_predict = True,
            per_device_eval_batch_size = 100,   
            dataloader_drop_last = False    
        )
        if model.training:
            raise Exception("Model is in training mode")
        model = Trainer(model=model, args=test_args)
        return model

    def get_predictions(self, model, dataset, checkpoint:str):
        output = self.predict(model,dataset,save=False)
        model_list = [
            'Maltehb/aelaectra-danish-electra-small-cased',
            'Maltehb/danish-bert-botxo',
            'xlm-roberta-large',
            'sentence-transformers/paraphrase-xlm-r-multilingual-v1',
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        ]
        if checkpoint in model_list:
            probs = [softmax(i).tolist() for i in output.predictions]
            return np.array(probs)[:,1]
        else: 
            logger.info("Model is not supported yet")

    def save_model(self, trainer, model_name: str, dir = '/interim/Sentence_scoring'):
        return trainer.save_model(f'{dir}/{model_name}')

if __name__ == "__main__":
    checkpoint = "Maltehb/danish-bert-botxo" # "Maltehb/-l-ctra-uncased"
    data_dir = ''

    train_args = TrainingArguments(
        output_dir='./results',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        do_train=True,
        do_eval=True
    )

    # Tokenized data
    train_val = DatasetDict.load(dir=data_dir, name='tokenized_data_train_val')
    test = DatasetDict.load(dir=data_dir, name='tokenized_test')

    # Sentence Scoring
    ss = SentenceScoring(checkpoint=checkpoint)

    # Fine-tune and evaluate
    logger.info("Fine-tuning pretrained model")
    trainer = ss.fine_tuning(train_val, train_args=None, freeze_layers=True)
    logger.info(ss.evaluate_model(trainer))

    # Predict on unseen test set
    output = ss.predict(trainer, test)
    logger.info(f"Prediction metrics: {output.metrics}")

    # Save model
    if output.metrics["test_accuracy"] > 0.65:
        logger.info("Saving model since its accuracy is above 65%")
        ss.save_model(trainer, model_name='fine_tuned_model')
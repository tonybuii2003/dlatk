# transformer_pipeline.py
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import dlatk.dlaConstants as dlac

class TransformerPipeline:
    def __init__(self, model_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

        # Get label mapping
        self.num_labels = self.config.num_labels
        self.id2label = getattr(self.config, "id2label", {i: f"class_{i}" for i in range(self.num_labels)})

    def extract_message_features(self, messages, batch_size=8):
        """
        Processes messages in batches to avoid CUDA Out-of-Memory errors.
        """
        all_logits = []
        all_probs = []
        
        dlac.warn(f"Processing {len(messages)} messages in batches of {batch_size}...")

        # Process in chunks
        for i in tqdm(range(0, len(messages), batch_size)):
            batch_texts = messages[i : i + batch_size]
            
            # Tokenize the current batch
            inputs = self.tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Move to CPU immediately to free up GPU memory
                all_logits.append(logits.cpu())
                all_probs.append(probs.cpu())
                
            # Optional: Explicitly clear cache if fragmentation is high
            # torch.cuda.empty_cache()

        # Concatenate all batches back into single numpy arrays
        final_logits = torch.cat(all_logits, dim=0).numpy()
        final_probs = torch.cat(all_probs, dim=0).numpy()

        return final_logits, final_probs

    def get_dlatk_rows(self, message_ids, logits, probs):
        """
        Formats the raw model output into the (feat, value, group_no) structure.
        """
        rows = []
        for i, msg_id in enumerate(message_ids):
            for class_idx in range(self.num_labels):
                rows.append({
                    'group_id': msg_id,
                    'feat': self.id2label[class_idx],
                    'value': float(logits[i][class_idx]),
                    'group_norm': float(probs[i][class_idx])
                })
        return rows
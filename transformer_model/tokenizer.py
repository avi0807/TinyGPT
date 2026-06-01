import re 
import pickle
from collections import Counter

  

class CharTokenizer:
    def __init__(self,text):
        self.chars =sorted(list(set(text)))     #create vocab

        self.char2id = {}                   #gives a number to every char
        for i,ch in enumerate(self.chars):
            self.char2id[ch] = i
        
        self.id2char = {}
        for i,ch in enumerate(self.chars):
            self.id2char[i]=ch 
        
        self.vocab_size=len(self.chars)
        
    def encode(self,text):
        return [self.char2id[ch] for ch in text]
        
    def decode(self,ids):
        return ''.join([self.id2char[i] for i in ids])
    

class BPE_tokenizer:
    def __init__(self,num_merges=1000):
        self.num_merges = num_merges   #how many merge operations to perform 
        self.merges = {}                    #storing learned merge rules
        self.vocab = {}                     #stores tokenized word frequency map
        self.specials = ["<|endoftext|>", "<|unk|>"]
        self._encode_cache = {}

    def get_vocab(self,text):
        words=text.strip().split()
        vocab=Counter()

        for word in words:
            chars=' '.join(list(word)) + ' </w>'     #changes "low"  to  "l o w </w>"
            vocab[chars] +=1                    #updates frequency
        return vocab
    
    def get_stats(self,vocab):
        pairs = Counter()           #frequency of tokens that come together

        for word,freq in vocab.items():
            symbols = word.split()

            for i in range(len(symbols)- 1):
                pairs[(symbols[i],symbols[i+1])] += freq        #tells which pair appears most often
        return pairs                                        #pair = ('l','o')
    
    def merge_vocab(self,pair,vocab):               # concatenate best pair ('l', 'o') to  'l o'  to "lo" in one token
        
        new_vocab={}

        for word , freq in vocab.items():
            tokens=word.split()

            i=0
            new_tokens=[]

            while i<len(tokens):
                if i<len(tokens)-1 and tokens[i]==pair[0] and tokens[i+1]==pair[1]:
                    new_tokens.append(pair[0]+pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i+=1
            new_word=' '.join(new_tokens)

            new_vocab[new_word]=freq
        return new_vocab  
   
    def train(self,text):
        vocab = self.get_vocab(text)

        for i in range(self.num_merges):
            pairs=self.get_stats(vocab)

            if len(pairs)==0:
                 print("stopped early at merge",i)
                 break


            best = max(pairs,key=pairs.get)
            vocab=self.merge_vocab(best,vocab)

            self.merges[best]=''.join(best)

        self.vocab=vocab
        print("Total merges learned:",len(self.merges))

    def encode_word(self,word):
        if word in self._encode_cache:
            return self._encode_cache[word]

        tokens=list(word)
        tokens.append("</w>")


        for pair,merged in self.merges.items():
            if len(tokens)==1:
                break
            
            i=0
            new_tokens=[]
            while i<len(tokens):
                if i<len(tokens)- 1 and (tokens[i],tokens[i+1]) == pair:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i+=1
            tokens=new_tokens

        self._encode_cache[word]=tokens
        return tokens
    
    def encode(self,text):              #token to id
        words = text.strip().split()
        tokens=[]
        for word in words:
            sub_tokens=self.encode_word(word)
            tokens.extend(sub_tokens)
        
        return [self.token2id.get(tok,self.unk_id) for tok in tokens]
    

    def build_token_mappings(self,text):             #mapping
        tokens=set()

        encoded_tokens = []

        words=text.strip().split()

        for word in words:
            encoded_tokens.extend(self.encode_word(word))
        
        
        tokens.update(encoded_tokens)

        self.token2id = {tok:i for i,tok in enumerate(self.specials)}
        offset=len(self.specials)

        for i,tok in enumerate(sorted(tokens - set(self.specials))):
            self.token2id[tok] = i + offset
        
        self.id2token = {i:tok for tok,i in self.token2id.items()}
        
        self.vocab_size = len(self.token2id)

        self.eos_id=self.token2id["<|endoftext|>"]
        self.unk_id=self.token2id["<|unk|>"]

    def decode(self,ids):                       #id to token
        # Filter out special tokens (eos/unk) so they don't appear verbatim in output.
        specials = set(getattr(self, "specials", []))
        tokens = [self.id2token[i] for i in ids if self.id2token[i] not in specials]
        text = ''.join(tokens).replace('</w>', ' ')
        return text
    def save(self,path):
        with open(path,"wb") as f:
            pickle.dump({
                "merges": self.merges,
                "token2id": self.token2id,
                "id2token": self.id2token,
                "vocab_size": self.vocab_size,
                "vocab": self.vocab,
                "specials":self.specials
            }, f)
        print("tokenizer weights saved to ",path)
    def load(self,path):
        with open(path,"rb") as f:
            data = pickle.load(f)
        self.merges = data["merges"]
        self.token2id = data["token2id"]
        self.id2token = data["id2token"]
        self.vocab_size = data["vocab_size"]
        self.vocab = data["vocab"]
        self.specials = data.get("specials", ["<|endoftext|>", "<|unk|>"])
        self.eos_id = self.token2id.get("<|endoftext|>")
        self.unk_id = self.token2id.get("<|unk|>")
        self._encode_cache = {}        
        print("Tokenizer loaded from", path)
    
    


class HFTokenizer:
    """
    Drop-in replacement for BPE_tokenizer backed by HuggingFace `tokenizers` (Rust).
    Same interface: train / encode / decode / save / load / vocab_size / eos_id / unk_id.
    Much faster, so CPU tokenization won't starve the GPU during training.
    """
    def __init__(self, vocab_size=16000):
        # I keep the same two specials as my BPE so the rest of the code is unchanged.
        self.target_vocab_size = vocab_size
        self.specials = ["<|endoftext|>", "<|unk|>"]
        self.tokenizer = None          # the underlying tokenizers.Tokenizer
        self.vocab_size = 0
        self.eos_id = None
        self.unk_id = None

    def train(self, text_iterator):
        # text_iterator: an iterable of strings (e.g. a generator over documents).
        # I use byte-level BPE so any unicode/whitespace/code survives round-trip.
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

        tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=self.target_vocab_size,
            special_tokens=self.specials,          # reserved at the lowest ids
        )
        tok.train_from_iterator(text_iterator, trainer=trainer)

        self.tokenizer = tok
        self._refresh_ids()
        print("HF tokenizer trained. vocab_size:", self.vocab_size)

    def _refresh_ids(self):
        # I cache the handy ids after train/load.
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.eos_id = self.tokenizer.token_to_id("<|endoftext|>")
        self.unk_id = self.tokenizer.token_to_id("<|unk|>")

    def encode(self, text):                         # str -> list[int]
        return self.tokenizer.encode(text).ids

    def decode(self, ids):                          # list[int] -> str
        # skip_special_tokens drops <|endoftext|>/<|unk|> from the output.
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def save(self, path):
        # tokenizers serializes to a single JSON file.
        self.tokenizer.save(path)
        print("HF tokenizer saved to", path)

    def load(self, path):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(path)
        self._refresh_ids()
        print("HF tokenizer loaded from", path)


if __name__ == "__main__":

    text = "hello world"

    tokenizer = CharTokenizer(text)

    encoded = tokenizer.encode("hello")
    decoded = tokenizer.decode(encoded)

    print("Encoded:", encoded)
    print("Decoded:", decoded)
    print("Vocab size:", tokenizer.vocab_size)




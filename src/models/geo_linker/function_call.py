from transformers import AutoTokenizer,AutoModelForCausalLM
import torch 
import re

class function_call():
    def __init__(self,model_id="hiieu/Meta-Llama-3-8B-Instruct-function-calling-json-mode"):
        self.model_id = model_id
        self.model_init()
        self.get_func_meta()


    def model_init(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,torch_dtype = torch.bfloat16,
            device_map="auto"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_func_meta(self):
        self.functions_metadata= [ 
                {
                    "type":"function",
                    "function":{
                        "name": "link_location",
                        "description" : "Given a text and some mentions, choose the right mentions and link them to the corresponding locations in OpenStreetMap. There could be multiple locations in the text",
                        "parameters" : {
                            "type" : "object" ,
                            "properties" : {
                                "address" : {
                                    "type" : "string",
                                    "description" : "the mention(s) of the locations in the text to link to OpenStreetMap"
                                }
                            },
                            "required" : ["address"]
                        }
                    }
                }
            ]
    def parse_output(self,output:str):
        pattern = r'"address":\s*"(.*?)"'
        arguements = re.findall(pattern, output)
        return arguements

    def get_func_arg(self, text:str, mentions:list):
        mentions_str = ", ".join(mentions)
        messages = [
            {
                "role": "system",
                "content": f"""You are a helpful assistant with access to the following function(s): 
        {str(self.functions_metadata)}

        To use these functions, respond with:
        <functioncall>{{"name": "function_name", "arguments": {{"arg_1": "value_1", "arg_2": "value_2", ...}}}}</functioncall>

        You may call multiple functions if needed, using the same format.

        Here is an example:
        Question: "What are the locations in '40 Essen Unfall 9 km Stau (Zeitverlust: mehr als eine Dreiviertelstunde)'? The mentions are: A40, Essen"
        <functioncall>{{"name": "link_location", "arguments": {{"address": "Essen"}}}}</functioncall>
        <functioncall>{{"name": "link_location", "arguments": {{"address": "A40 Essen"}}}}</functioncall>
        """
            },
            {
                "role": "user",
                "content": f"What are the locations in {text}? The mentions are: {mentions_str}"
            }
        ]


        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generatio_prompt =True ,
            return_tensors = "pt"
        ).to(self.model.device)
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        attention_mask = torch.ones_like(input_ids, device=self.model.device)
        outputs = self.model.generate(
            input_ids,
            attention_mask = attention_mask,
            max_new_tokens = 256,
            eos_token_id = terminators,
            do_sample = True,
            temperature = 0.6,
            top_p = 0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        output = self.tokenizer.decode(response,skip_special_tokens=True)
        print(f"Output: {output}")
        return self.parse_output(output)

if __name__ == "__main__":
    text = "Kreis Biberach, K7588 zwischen Daugendorf und Unlingen in beiden Richtungen Gefahr durch Hochwasser, Verbindungsfahrbahn gesperrt, eine \u00f6rtliche Umleitung ist eingerichtet, bis 19.04.2016 17:00 Uhr\n"
    mentions = ["Tiefgarage Kreissparkasse Biberach", "Daugendorfer", "Unlingen"]
    test = function_call()
    print(test.get_func_arg(text, mentions))
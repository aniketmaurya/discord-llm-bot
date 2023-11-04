from typing import Any
from loguru import logger
from pathlib import Path
from llama_cpp import Llama

DEFAULT_MODEL = str(Path.home()/"mistral-7b-v0.1.Q4_K_M.gguf")

class LlamaCppLLM:
    def __init__(self, model_path=DEFAULT_MODEL, main_gpu=1, n_ctx=1028, lazy=False, *args, **kwargs):
        if lazy:
            self.llm = None
        else:
            self.llm = Llama(model_path=model_path, main_gpu=main_gpu, n_ctx=n_ctx, verbose=False, *args, **kwargs)
        self.n_ctx= n_ctx
        self.model_path = model_path
        self.main_gpu = main_gpu
        self.args = args
        self.kwargs = kwargs


    def __call__(self, query:str, document: str, *args: Any, **kwargs: Any) -> Any:
        if self.llm is None:
            logger.info("Initializing model")
            self.llm = Llama(model_path=self.model_path, main_gpu=self.main_gpu, n_ctx=self.n_ctx, verbose=False, *self.args, **self.kwargs)
            
        prompt = f"""Answer the given question based on the context. If you don't know the answer then respond with I don't know.
        Context: {document[:self.n_ctx-50]}
        ----
        Q: {query}
        A:"""

        result =  self.llm(prompt=prompt, *args, **kwargs)
        logger.info(result)
        return result["choices"][0]["text"]

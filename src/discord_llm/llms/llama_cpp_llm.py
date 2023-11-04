from typing import Any
import os
from loguru import logger
from pathlib import Path
from llama_cpp import Llama


# https://huggingface.co/TheBloke/OpenHermes-2-Mistral-7B-GGUF
DEFAULT_MODEL = "openhermes-2-mistral-7b.Q6_K.gguf"


class LlamaCppLLM:
    def __init__(
        self,
        model_path=DEFAULT_MODEL,
        main_gpu=1,
        n_ctx=1028,
        lazy=False,
        *args,
        **kwargs,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found.")
        if lazy:
            self.llm = None
        else:
            self.llm = Llama(
                model_path=model_path,
                main_gpu=main_gpu,
                n_ctx=n_ctx,
                verbose=False,
                *args,
                **kwargs,
            )
        self.n_ctx = n_ctx
        self.model_path = model_path
        self.main_gpu = main_gpu
        self.args = args
        self.kwargs = kwargs

    def __call__(self, query: str, document: str, *args: Any, **kwargs: Any) -> Any:
        if self.llm is None:
            logger.info("Initializing model")
            self.llm = Llama(
                model_path=self.model_path,
                main_gpu=self.main_gpu,
                n_ctx=self.n_ctx,
                verbose=False,
                *self.args,
                **self.kwargs,
            )

        prompt = f"""Answer the given question based on the given context. If you don't know the answer then respond with "I couldn't figure that out", don't make up answer and avoid wrong answers.
        ```{document[:self.n_ctx-50]}```
        ----
        Question: {query}
        Answer:"""

        result = self.llm(prompt=prompt, *args, **kwargs)
        logger.info(result)
        return result["choices"][0]["text"]

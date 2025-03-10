import json
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sse_starlette import EventSourceResponse
from typing import List, Tuple

from llmtuner.tuner import get_infer_args
from llmtuner.extras.misc import torch_gc
from llmtuner.chat.stream_chat import ChatModel
from llmtuner.api.protocol import (
    ModelCard,
    ModelList,
    ChatMessage,
    DeltaMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionResponseUsage
)


@asynccontextmanager
async def lifespan(app: FastAPI): # collects GPU memory
    yield
    torch_gc()


def create_app():
    chat_model = ChatModel(*get_infer_args())

    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/v1/models", response_model=ModelList)
    async def list_models():
        model_card = ModelCard(id="gpt-3.5-turbo")
        return ModelList(data=[model_card])

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def create_chat_completion(request: ChatCompletionRequest):
        if request.messages[-1].role != "user":
            raise HTTPException(status_code=400, detail="Invalid request")
        query = request.messages[-1].content

        prev_messages = request.messages[:-1]
        if len(prev_messages) > 0 and prev_messages[0].role == "system":
            prefix = prev_messages.pop(0).content
        else:
            prefix = None

        history = []
        if len(prev_messages) % 2 == 0:
            for i in range(0, len(prev_messages), 2):
                if prev_messages[i].role == "user" and prev_messages[i+1].role == "assistant":
                    history.append([prev_messages[i].content, prev_messages[i+1].content])

        if request.stream:
            generate = predict(query, history, prefix, request)
            return EventSourceResponse(generate, media_type="text/event-stream")

        response, (prompt_length, response_length) = chat_model.chat(
            query, history, prefix, temperature=request.temperature, top_p=request.top_p, max_new_tokens=request.max_tokens
        )

        usage = ChatCompletionResponseUsage(
            prompt_tokens=prompt_length,
            completion_tokens=response_length,
            total_tokens=prompt_length+response_length
        )

        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=response),
            finish_reason="stop"
        )

        return ChatCompletionResponse(model=request.model, choices=[choice_data], usage=usage, object="chat.completion")

    async def predict(query: str, history: List[Tuple[str, str]], prefix: str, request: ChatCompletionRequest):
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(role="assistant"),
            finish_reason=None
        )
        chunk = ChatCompletionStreamResponse(model=request.model, choices=[choice_data], object="chat.completion.chunk")
        yield json.dumps(chunk, ensure_ascii=False)

        for new_text in chat_model.stream_chat(
            query, history, prefix, temperature=request.temperature, top_p=request.top_p, max_new_tokens=request.max_tokens
        ):
            if len(new_text) == 0:
                continue

            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content=new_text),
                finish_reason=None
            )
            chunk = ChatCompletionStreamResponse(model=request.model, choices=[choice_data], object="chat.completion.chunk")
            yield json.dumps(chunk, ensure_ascii=False)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(),
            finish_reason="stop"
        )
        chunk = ChatCompletionStreamResponse(model=request.model, choices=[choice_data], object="chat.completion.chunk")
        yield json.dumps(chunk, ensure_ascii=False)
        yield "[DONE]"

    return app


if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)

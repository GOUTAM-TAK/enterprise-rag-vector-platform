from nemoguardrails.actions import action

@action
def content_safety_check(text: str) -> dict:
    # Call NVIDIA NIM safety model here
    return {"allowed": True}

@action
def topic_safety_check(text: str) -> dict:
    # Call NVIDIA topic-control model here
    return {
        "allowed": True
    }
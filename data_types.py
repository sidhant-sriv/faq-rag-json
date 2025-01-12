from pydantic import BaseModel

class FAQ(BaseModel):
    question: str
    tags: list[str]
    answer: str

    def to_text(self):
        return f"Question: {self.question}\nTags: {', '.join(self.tags)}\nAnswer: {self.answer}"

class FAQList(BaseModel):
    faqs: list[FAQ]
    count: int

    def __init__(self, faqs: list[FAQ]):
        self.faqs = faqs
        self.count = len(faqs)

    def to_text(self):
        return "\n\n".join(faq.to_text() for faq in self.faqs)
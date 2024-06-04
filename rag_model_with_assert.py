import dspy
from dsp.utils import deduplicate
from dspy.primitives.assertions import assert_transform_module, backtrack_handler
from dspy.predict import Retry
from utils import *

class GenerateSearchQueryFromChatHist(dspy.Signature):
    """Please using Vietnamese language to complete/paraphrase the User question based on chat history. Remember dont try to answer, rewrite more clearly the content of User question"""

    context = dspy.InputField(desc="contain chat history")
    question = dspy.InputField(desc="User question")
    query = dspy.OutputField()

class GenerateAnswer(dspy.Signature):
    """Answer questions, please using Vietnamese language."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 1225 words")

class GenerateSearchQuery(dspy.Signature):
    """Please using Vietnamese language to Write a simple search query that will help answer a complex question"""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()


class GenerateCitedParagraph(dspy.Signature):
    """Please using Vietnamese language to Generate a paragraph with citations in format text ... [x]. Hãy sử dụng tiếng Việt."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    paragraph = dspy.OutputField(desc="includes citations")

class GenerateCitedParagraph_with_hist(dspy.Signature):
    """Based on chat history to understand the question,  Based on relevant facts to Answer questions. Please using Vietnamese language to Generate a paragraph with citations in format text ... [x]... [y], dont use this format text ... [x, y]. Hãy sử dụng tiếng Việt."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField(desc="question")
    chat = dspy.InputField(desc="contain chat history")
    paragraph = dspy.OutputField(desc="includes citations")


class GenerateAnswer_with_hist(dspy.Signature):
    """Based on chat history to understand the question , Based on relevant facts to Answer questions, please using Vietnamese language."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    chat = dspy.InputField(desc="contain chat history")

    answer = dspy.OutputField(desc="often between 1 and 1225 words")

# citation with assertion
class LongFormQAWithAssertions(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=3,retriever_q = None,retriever_f = None):
        super().__init__()
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retriever_q = retriever_q
        self.retriever_f = retriever_f
        self.generate_cited_paragraph_with_hist = dspy.ChainOfThought(GenerateCitedParagraph_with_hist)
        self.generate_cited_paragraph = dspy.ChainOfThought(GenerateCitedParagraph)
        self.verify_answer = dspy.ChainOfThought(CheckQuestioAnswer_Related)
        self.max_hops = max_hops
    
    def forward(self, question,chat_hist):
        context = []
        prev_queries = [question]
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            # dspy.Suggest(
            #     validate_query_distinction_local(prev_queries, query),
            #     "Query should be distinct from: "
            #     + "; ".join(f"{i+1}) {q}" for i, q in enumerate(prev_queries)),
            # )

            passages_1 = self.retriever_q(query)
            passages_q = [s['metadatas']['full'] for s in passages_1]
            passages_2 = self.retriever_f(query)
            passages_f = [s['metadatas']['full'] for s in passages_2]
            context = deduplicate(context + passages_q + passages_f)

        # if len(chat_hist)>1:
        #     pred = self.generate_cited_paragraph_with_hist(context=context, question=question,chat = chat_hist )
        #     print("chat_hist",chat_hist)
        #     print("pred",pred.paragraph)
        # else:
        #     pred = self.generate_cited_paragraph(context=context, question=question )
        
        pred = self.generate_cited_paragraph(context=context, question=question )

        pred = dspy.Prediction(context=context, paragraph=pred.paragraph)
        dspy.Suggest(citations_check(pred.paragraph), f"Make sure every 1-2 sentences has citations. If any 1-2 sentences lack citations, add them in 'text... [x].' format.", target_module=GenerateCitedParagraph)

        _, unfaithful_outputs = citation_faithfulness(None, pred, None)
        if unfaithful_outputs:
            unfaithful_pairs = [(output['text'], output['context']) for output in unfaithful_outputs]
            for _, context in unfaithful_pairs:
                dspy.Suggest(len(unfaithful_pairs) == 0, f"Make sure your output is based on the following context: '{context}'.", target_module=GenerateCitedParagraph)
        else:
            return pred
            # return pred
        ## verify answer
        # dspy.Suggest(self.verify_answer(question = question, answer = pred ), f"Make sure your answer is related to question: '{question}'.", target_module=GenerateCitedParagraph)

        
        return pred
    
### simple multihop without citation
class SimplifiedBaleen(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2,retriever_q = None,retriever_f = None):
        super().__init__()

        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]

        self.retriever_q = retriever_q
        self.retriever_f = retriever_f

        self.generate_answer = dspy.ChainOfThought(GenerateAnswer_with_hist)
        self.max_hops = max_hops
    
    def forward(self, question,chat_hist):
        context = []
        
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages_1 = self.retriever_q(query)
            passages_q = [s['metadatas']['full'] for s in passages_1]
            passages_2 = self.retriever_f(query)
            passages_f = [s['metadatas']['full'] for s in passages_2]
            context = deduplicate(context + passages_q + passages_f)

        pred = self.generate_answer(context=context, question=question, chat = chat_hist)
        return dspy.Prediction(context=context, answer=pred.answer)
#### additional utils
class CheckCitationFaithfulness(dspy.Signature):
    """Verify that the text is based on the provided context."""
    context = dspy.InputField(desc="may contain relevant facts")
    text = dspy.InputField(desc="between 1 to 2 sentences")
    faithfulness = dspy.OutputField(desc="boolean indicating if text is faithful to context")

class CheckQuestioAnswer_Related(dspy.Signature):
    """Verify the answer must be related to question."""
    question = dspy.InputField(desc="Question")
    answer = dspy.InputField(desc="Answer")
    faithfulness = dspy.OutputField(desc="boolean indicating if answer is realted to question")

def citation_faithfulness(example, pred, trace):
    paragraph, context = pred.paragraph, pred.context
    # print("paragraph, context",paragraph, context)
    citation_dict = extract_text_by_citation(paragraph)
    # print("citation_dict",citation_dict)
    # print(context[0])
    if not citation_dict:
        return False, None
    # context_dict = {str(i): context[i].split(' | ')[1] for i in range(len(context))}
    context_dict = {str(i): context[i] for i in range(len(context))}

    faithfulness_results = []
    unfaithful_citations = []
    check_citation_faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
    for citation_num, texts in citation_dict.items():
        if citation_num not in context_dict:
            continue
        current_context = context_dict[citation_num]
        for text in texts:
            try:
                result = check_citation_faithfulness(context=current_context, text=text)
                is_faithful = result.faithfulness.lower() == 'true'
                faithfulness_results.append(is_faithful)
                if not is_faithful:
                    unfaithful_citations.append({'paragraph': paragraph, 'text': text, 'context': current_context})
            except ValueError as e:
                faithfulness_results.append(False)
                unfaithful_citations.append({'paragraph': paragraph, 'text': text, 'error': str(e)})
    final_faithfulness = all(faithfulness_results)
    if not faithfulness_results:
        return False, None
    return final_faithfulness, unfaithful_citations
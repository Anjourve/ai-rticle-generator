import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter


st.set_page_config( page_title="Kalungi Ai-rticle", layout="wide")
st.header(":blue[Please provide us with the article you'd like to work with.]")

openai_api_key_input = st.text_area(label = "", placeholder = "Please enter the API OpenAI.", key = "openai_api_key_input")
article = st.text_area(label = "", placeholder = "Please enter the text or URL of the article here.", key = "article_imput")

def get_true_or_false_article(article):
    prompt_temp = PromptTemplate(input_variables = ["dataarticle"], template = """ for this text = {dataarticle}
                                                                               - If the text is a URL, it returns "FALSE"; otherwise, it returns "TRUE".
                                                                               """)
    prompt_value = prompt_temp.format(dataarticle= article)

    llm_openai = OpenAI(model_name = "gpt-3.5-turbo-16k", temperature=0, openai_api_key = openai_api_key_input)
    respuesta_openai = llm_openai(prompt_value)
    return respuesta_openai


how_to_describe_tone ="""
1. Pace: The speed at which the story progresses, which can create suspense or relaxation.
2. Mood: The overall feeling or atmosphere that a piece of writing creates for the reader.
3. Voice: The unique style or point of view of the author.
4. Diction: The choice of words and phrases in a piece of writing.
5. Syntax: The arrangement of words and phrases to create well-formed sentences.
6. Imagery: The use of descriptive language to create visual representations of actions, objects, or ideas.
7. Theme: The underlying message or main idea that the writer wants to convey.
8. Perspective: The angle of considering things, which shows the opinion or feelings of the individuals involved in a situation.
9. Irony: The use of words to convey a meaning that is the opposite of its literal meaning.
10. Humor: The quality of being amusing or comic, especially as expressed in literature.
11. Sarcasm: The use of irony to mock or convey contempt.
12. Sentimentality: The quality of being excessively sentimental or emotional.
13. Formality: The level of seriousness or informality in the language used.
14. Rhythm: The pattern of stressed and unstressed syllables in a line of writing.
15. Figurative Language: The use of words or expressions with a meaning that is different from the literal interpretation.
16. Connotation: The emotional or cultural association with a word beyond its dictionary definition.
17. Allusion: A reference to a well-known person, place, event, literary work, or work of art.
18. Symbolism: The use of symbols to represent ideas or qualities.
19. Foreshadowing: The use of hints or clues to suggest events that will occur later in the story.
20. Allegory: A story, poem, or picture that can be interpreted to reveal a hidden meaning, typically a moral or political one.
"""

def get_authors_tone_description(how_to_describe_tone, blogarticle):
    template = """
        You are an AI Bot that is very good at generating writing in a similar tone as examples.
        Be opinionated and have an active voice.
        Take a strong stance with your response.

        % HOW TO DESCRIBE TONE
        {how_to_describe_tone}

        % START OF EXAMPLES
        {blog}
        % END OF EXAMPLES

        List out the tone qualities of the example above
        """

    prompt = PromptTemplate(
        input_variables=["how_to_describe_tone", "blog"],
        template=template,
    )

    final_prompt = prompt.format(how_to_describe_tone=how_to_describe_tone, blog=blogarticle)

    tonearticle = llm.predict(final_prompt)

    return tonearticle

def get_tone_author(blogarticle):

    prompt_temp = PromptTemplate(input_variables = ["docstext"], template = """ You are an AI Bot that is very good at identifying authors, public figures, or writers whos style matches a piece of text. Be opinionated and have an active voice. Take a firm stance with your response.
                                                                                Respond with only the full name (list up to 4 full name if necessary) of the author, public figure, or writer that sounds most closely resemble following text:
                                                                                {docstext}""")
    prompt_value = prompt_temp.format(docstext = blogarticle)
    llm_openai = OpenAI(model_name = "gpt-3.5-turbo-16k", temperature=0, openai_api_key = openai_api_key_input)
    toneauthor = llm_openai(prompt_value)
    return toneauthor

if article:
  articleevaluated = get_true_or_false_article(article)

  if articleevaluated == "TRUE":
    blogarticle = article
    st.write("Su Articulo es este:\n\n"+blogarticle)
    st.write("---\n\n")
    answertone = get_tone_article(blogarticle)
    st.write("Tono del articulo:\n\n"+answertone)
    st.write("---\n\n")
    answertoneauthor = get_tone_author(blogarticle)
    st.write("Author:\n\n"+answertoneauthor)
  else:
    def get_datablog(article):
        articleurl = f"{article}"
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:80.0) Gecko/20100101 Firefox/80.0'}
        loader = UnstructuredURLLoader(urls=[articleurl], headers=headers, ssl_verify=False)
        return loader.load()

    def get_article(article):
        dataarticleurl = get_datablog(article)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 0)
        docs = text_splitter.split_documents(dataarticleurl)

        prompt_temp = PromptTemplate(input_variables = ["docstext"], template = """ for this data = {docstext}
                                                                                - Extract the title of the article and all the content of the article (Leave a line between paragraphs every time you come across \n\n), excluding things that are unrelated, such as footnotes, buttons, information that is not relevant to the article or Or information about seeing any other articles made by the author.
                                                                                """)
        prompt_value = prompt_temp.format(docstext= docs)
        llm_openai = OpenAI(model_name = "gpt-3.5-turbo-16k", temperature=0, openai_api_key = openai_api_key_input)
        respuesta_openai = llm_openai(prompt_value)
        return respuesta_openai
    blogarticle = get_article(article)
    st.write("Su Articulo es este:\n\n"+blogarticle)
    st.write("---\n\n")
    answertone = get_tone_article(blogarticle)
    st.write("Tono del articulo:\n\n"+answertone)
    st.write("---\n\n")
    answertoneauthor = get_tone_author(blogarticle)
    st.write("Author:\n\n"+answertoneauthor)

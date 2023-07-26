import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter


st.set_page_config( page_title="Kalungi Ai-rticle", layout="wide")
st.header(":blue[Please provide us with the article you'd like to work with.]")

article = st.text_area(label = "", placeholder = "Please enter the text or URL of the article here.", key = "article_imput")

def get_true_or_false_article(article):
    prompt_temp = PromptTemplate(input_variables = ["dataarticle"], template = """ for this text = {dataarticle}
                                                                               - If the text is a URL, it returns "FALSE"; otherwise, it returns "TRUE".
                                                                               """)
    prompt_value = prompt_temp.format(dataarticle= article)

    llm_openai = OpenAI(model_name = "gpt-3.5-turbo-16k", temperature=0, openai_api_key = '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')
    respuesta_openai = llm_openai(prompt_value)
    return respuesta_openai

def get_tone_article(blogarticle):
    selecttone = """1. Pace: The speed at which the story unfolds and events occur.
                    2. Mood: The overall emotional atmosphere or feeling of the piece.
                    3. Tone: The author's attitude towards the subject matter or characters.
                    4. Voice: The unique style and personality of the author as it comes through in the writing.
                    5. Diction: The choice of words and phrases used by the author.
                    6. Syntax: The arrangement of words and phrases to create well-formed sentences.
                    7. Imagery: The use of vivid and descriptive language to create mental images for the reader.
                    8. Theme: The central idea or message of the piece.
                    9. Point of View: The perspective from which the story is told (first person, third person, etc.).
                    10. Structure: The organization and arrangement of the piece, including its chapters, sections, or stanzas.
                    11. Dialogue: The conversations between characters in the piece.
                    12. Characterization: The way the author presents and develops characters in the story.
                    13. Setting: The time and place in which the story takes place.
                    14. Foreshadowing: The use of hints or clues to suggest future events in the story.
                    15. Irony: The use of words or situations to convey a meaning that is opposite of its literal meaning.
                    16. Symbolism: The use of objects, characters, or events to represent abstract ideas or concepts.
                    17. Allusion: A reference to another work of literature, person, or event within the piece.
                    18. Conflict: The struggle between opposing forces or characters in the story.
                    19. Suspense: The tension or excitement created by uncertainty about what will happen next in the story.
                    20. Climax: The turning point or most intense moment in the story.
                    21. Resolution: The conclusion of the story, where conflicts are resolved and loose ends are tied up."""

    prompt_temp = PromptTemplate(input_variables = ["docstext","tones"], template = """ You are an AI bot that is very good at identifying the types of tones in web articles. Be opinionated and have an active voice. Take a firm stance with your response.
                                                                                        According to the following tones:
                                                                                        {tones}

                                                                                        Describe the tone of this web article:
                                                                                        {docstext}""")
    prompt_value = prompt_temp.format(docstext = blogarticle, tones = selecttone)
    llm_openai = OpenAI(model_name = "gpt-3.5-turbo-16k", temperature=0, openai_api_key = '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')
    tonearticle = llm_openai(prompt_value)
    return tonearticle

def get_tone_author(blogarticle):

    prompt_temp = PromptTemplate(input_variables = ["docstext"], template = """ You are an AI Bot that is very good at identifying authors, public figures, or writers whos style matches a piece of text. Be opinionated and have an active voice. Take a firm stance with your response.
                                                                                Respond with only the full name (list up to 4 full name if necessary) of the author, public figure, or writer that sounds most closely resemble following text:
                                                                                {docstext}""")
    prompt_value = prompt_temp.format(docstext = blogarticle)
    llm_openai = OpenAI(model_name = "gpt-3.5-turbo-16k", temperature=0, openai_api_key = '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')
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
        llm_openai = OpenAI(model_name = "gpt-3.5-turbo-16k", temperature=0, openai_api_key = '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')
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

import streamlit as st
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md


st.set_page_config( page_title="Kalungi Ai-rticle", layout="wide")
#st.header(":blue[Please provide us with the article you'd like to work with.]")
openai_api_key_input = st.sidebar.text_input('OpenAI API Key', type='password')

#article = st.text_area(label = "", placeholder = "Please enter the text or URL of the article here.", key = "article_imput")


def get_true_or_false_article(article):
    prompt_temp = PromptTemplate(input_variables = ["dataarticle"], template = """ for this text = {dataarticle}
                                                                               - If the text is a URL, it returns "FALSE"; otherwise, it returns "TRUE".
                                                                               """)
    prompt_value = prompt_temp.format(dataarticle= article)
    llm_openai = OpenAI(model_name = "gpt-4", temperature=.7, openai_api_key = openai_api_key_input)
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
    llm_openai = OpenAI(model_name = "gpt-4", temperature=.7, openai_api_key = openai_api_key_input)
    tonearticle = llm_openai.predict(final_prompt)

    return tonearticle

def get_similar_public_figures(blogarticle):
    template = """
    You are an AI Bot that is very good at identifying authors, public figures, or writers whos style matches a piece of text
    Your goal is to identify which authors, public figures, or writers sound most similar to the text below

    % START EXAMPLES
    {examples}
    % END EXAMPLES

    Which authors (list up to 4 if necessary) most closely resemble the examples above? Only respond with the names separated by commas
    """

    prompt = PromptTemplate(
        input_variables=["examples"],
        template=template,
    )

    # Using the short list of examples so save on tokens and (hopefully) the top tweets
    final_prompt = prompt.format(examples=blogarticle)
    llm_openai = OpenAI(model_name = "gpt-4", temperature=.7, openai_api_key = openai_api_key_input)
    authors = llm_openai.predict(final_prompt)
    return authors
    
#def get_datablog(article):
#    articleurl = f"{article}"
#    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:80.0) Gecko/20100101 Firefox/80.0'}
#    loader = UnstructuredURLLoader(urls=[articleurl], headers=headers, ssl_verify=False)
#    return loader.load()
    
def get_datablog(article):

    # Doing a try in case it doesn't work
    try:
        response = requests.get(article)
    except:
        # In case it doesn't work
        print ("Whoops, error")
        return

    # Put your response in a beautiful soup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Get your text
    text = soup.get_text()

    # Convert your html to markdown. This reduces tokens and noise
    text = md(text)

    template="""

    % INSTRUCTIONS
     - You are an AI Bot that is very good at extracting blogs from websites.
     - You will receive raw text with markdowns from a webpage
     - Parse through the data and return only the blog post including both title and body
     - Do not answer anything other than the blog post

    % Website Data:
    {text}

    % Your Output
    """

    prompt = PromptTemplate(
    input_variables=["text"],
    template=template,
    )

    final_prompt = prompt.format(text=text)
    llm_openai = OpenAI(model_name = "gpt-4", temperature=.7, openai_api_key = openai_api_key_input)
    datablog = llm_openai.predict(final_prompt)
    
    return datablog

def header_and_title_tags(article):
    template="""
    Act as blog formatter GPT
    Take on the following blog and add header and title tags where you think appropiate;.

    {text}

    % Your Output
    """
    prompt = PromptTemplate(
        input_variables=["text"],
        template=template,
    )    

    final_prompt = prompt.format(text=article)

    llm_openai = OpenAI(model_name = "gpt-4", temperature=.7, openai_api_key = openai_api_key_input)
    output = llm_openai.predict(final_prompt)
    
    return output

def generate_outline(blogarticle):
    template ="""

    % INSTRUCTIONS
     - As an experienced data scientist and technical writer.
     - Generate an outline for this blog.
     - Take the <h1>, <h2>, and <h3> to generate the outline.
     - Do not answer anything other than the outline

    % blog:
    {topic}.

    % Your Output
    """
    
    prompt = PromptTemplate(input_variables=['topic'], template=template)

    prompt_query = prompt.format(topic=blogarticle)
    llm_openai = OpenAI(model_name = "gpt-4", temperature=.7, openai_api_key = openai_api_key_input)
    response_outline = llm_openai(prompt_query)

    return response_outline

def generate_new_outline(responseoutline):
    template ="""

    % INSTRUCTIONS
     - As an experienced data scientist and technical writer. 
     - By imitating one of these authors {authors}, it generates a new outline from the previous outline without them being similar.
     - Do not mention authors.
     - Do not answer anything other than the outline.

    % last outline:
    {topic}.

    % Your Output
    """
    
    prompt = PromptTemplate(input_variables=['topic', 'authors'], template=template)

    prompt_query = prompt.format(topic=responseoutline,authors=answertoneauthor)
    llm_openai = OpenAI(model_name = "gpt-4", temperature=.7, openai_api_key = openai_api_key_input)
    new_response_outline = llm_openai(prompt_query)

    return new_response_outline

def generate_new_outline(template, answertone, answertoneauthor, new_response_outline):
    template = """
    % INSTRUCTIONS
     - You are an AI Bot that is very good at mimicking an author writing style.
     - Your goal is to write content with the tone that is described below.
     - Do not go outside the tone instructions below

    % Mimic These Authors:
    {answertoneauthor}

    % Description of the authors tone:
    {answertone}

    % outline to use
    {new_response_outline}
    % End of outline to use

    % YOUR TASK
    1st - Write an article, following the outline to be used.
    2nd - Take on the following article and add header and title tags where you think appropiate as if you were the author described above;.
    """
    method_4_prompt_template = PromptTemplate(
    input_variables=["answertoneauthor", "answertone", "new_response_outline"],
    template=template,
    )
    final_prompt = method_4_prompt_template.format(answertoneauthor=answertoneauthor,
                                                   answertone=answertone,
                                                   new_response_outline=new_response_outline)
    llm_openai = OpenAI(model_name = "gpt-4", temperature=.7, openai_api_key = openai_api_key_input)
    new_article = llm_openai.predict(final_prompt)

    return new_article
    
#def get_article(article):
#    dataarticleurl = get_datablog(article)
#    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 0)
#    docs = text_splitter.split_documents(dataarticleurl)

#    prompt_temp = PromptTemplate(input_variables = ["docstext"], template = """ for this data = {docstext}
#                                                                                - Extract the title of the article and all the content of the article (Leave a line between paragraphs every time you come across \n\n), excluding things that are unrelated, such as footnotes, buttons, information that is not relevant to the article or Or information about seeing any other articles made by the author.
#                                                                                """)
#    prompt_value = prompt_temp.format(docstext= docs)
#    respuesta_openai = llm_openai(prompt_value)
#    return respuesta_openai

with st.form('myform'):
  article = st.text_input('Please enter the text or URL of the article here.:', '')
  submitted = st.form_submit_button('Submit')
  if not openai_api_key_input.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  if submitted and openai_api_key_input.startswith('sk-'):
    
    if article:
      articleevaluated = get_true_or_false_article(article)

      if articleevaluated == "TRUE":
        blogarticle = header_and_title_tags(article)
        st.info("Article:\n\n"+blogarticle)
        st.write("---\n\n")
      else:
        data = get_datablog(article)
        blogarticle = header_and_title_tags(data)
        st.info("Article:\n\n"+blogarticle)
        st.write("---\n\n")
      answertone = get_authors_tone_description(how_to_describe_tone, blogarticle)
      st.info("Tone Description:\n\n"+answertone)
      st.write("---\n\n")
      answertoneauthor = get_similar_public_figures(blogarticle)
      st.info("Author:\n\n"+answertoneauthor)
      st.write("---\n\n")
      responseoutline = generate_outline(blogarticle)
      st.info("Outline:\n\n"+responseoutline)      
      st.write("---\n\n")
      new_response_outline = generate_new_outline(responseoutline)
      st.info("New Outline:\n\n"+new_response_outline)
      st.write("---\n\n")
      new_article = generate_new_outline(template, answertone, answertoneauthor, new_response_outline)
      st.info("New Article:\n\n"+new_article)
      

import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';

const PROMPT = (context: string, history: string, question: string) => {
  return `You are a helpful AI assistant. Use the following pieces of context & chat history to answer the question at the end. If you don't know the answer, just say you don't know. DO NOT try to make up an answer. DO NOT try to make up things like Dates, Names, Places, etc.
If the question is not related to the context, politely respond that you don't know and ask if there is anything else you can help with related to Collect chat.
  
Context:
${context}
  
If the answer is not included, follow the rules below:
1. If the question is a greeting, respond back politely.
2. If the qestion is unrelated to context, politely respond that you don't know and ask if there is anything else you can help with related to Collect chat.
3. If the question is a statement or a fact, ALWAYS respond that you can only answer questions that are related to Collect chat.

Make sure the answer is in markdown format. Add line breaks when needed. Use bullet points if needed. Use bold, italics, and links if needed. Hyperlink URLs if possible. If the answer is a code snippet, use the code block markdown.

Provided below is a history of the conversation. You may also make use of the conversation history for additional context for the question at the end.

Conversation History:
${history}

Question: """
${question}
"""

Helpful answer in markdown:`;
};

export const generateChat = async (
  query: string,
  history: [string, string][],
  vectorStore: PineconeStore,
) => {
  const embeddings = new OpenAIEmbeddings();
  const results = await vectorStore.similaritySearchVectorWithScore(
    await embeddings.embedQuery(query),
    3,
  );
  //   .then((res) => res.map(([result, score]) => ({
  //     ...result,
  //     metadata: { ...result.metadata, score },
  //   })));

  const data = results.map(([result, score]) => ({
    ...result,
    metadata: { ...result.metadata, score },
  }));
  const context = data.map((item) => `${item.pageContent}\n`).join('\n');

  const formattedHistory = history
    ?.map(
      ([humanMessage, aiMessage]) =>
        `Human: ${humanMessage}, AI Assistant: ${aiMessage}, `,
    )
    .join('');

  const prompt = PROMPT(context, formattedHistory, query);
  console.log('🚀 ~ prompt:', prompt);
  const body = {
    model: 'gpt-3.5-turbo',
    temperature: 0,
    messages: [
      //   { role: 'user', content: query },
      { role: 'user', content: prompt },
    ],
  };

  const options = {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      'content-type': 'application/json',
    },
    body: JSON.stringify(body),
  };

  const response = await (
    await fetch('https://api.openai.com/v1/chat/completions', options)
  ).json();

  return {
    text: response.choices[0].message.content,
    sourceDocuments: data,
  };
};

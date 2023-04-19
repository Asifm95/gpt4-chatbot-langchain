import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { pinecone } from '@/utils/pinecone-client';
import { PINECONE_INDEX_NAME, PINECONE_NAME_SPACE } from '@/config/pinecone';
import { MongoClient } from 'mongodb';
import { ObjectId } from 'mongodb';

/* Name of directory to retrieve your files from */

export const run = async () => {
  try {
    console.log('connecting to database...');
    if (!process.env.MONGO_URI) {
      throw new Error('Missing MongoDB URI in .env file');
    }
    // @ts-ignore
    const client = await MongoClient.connect(process.env.MONGO_URI);
    console.log('connected to database');

    /*load documents from db */
    const coll = client.db('hc').collection('pages');
    const cursor = coll.find({
      centerId: new ObjectId(PINECONE_NAME_SPACE),
    });
    const result = await cursor.toArray();

    /* Split text into chunks */
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 2000,
      chunkOverlap: 0,
    });

    console.log('creating vector store...');
    /*create and store the embeddings in the vectorStore*/
    const embeddings = new OpenAIEmbeddings();

    const pineconeStore = await PineconeStore.fromExistingIndex(embeddings, {
      pineconeIndex: pinecone.Index(PINECONE_INDEX_NAME),
      namespace: 'hc-collectchat',
    });

    let docs: any[] = [];
    for (const page of result) {
      if (!page.title || !page.text) continue;

      const textArrays = await textSplitter.splitText(`#${page.title}
        
      =====================

      ${page.text}`);

      const docOutput = textArrays.map((text) => ({
        pageContent: text,
        metadata: {
          id: page._id,
          title: page.title,
          tags: page.tags,
          slugId: page.slugId,
        },
      }));
      docs = [...docs, ...docOutput];
    }
    console.log('docs', docs);
    console.log('calling addDocument ingesting data...');
    await pineconeStore.addDocuments(docs).then((res) => {
      console.log('res: completed', res);
    });
  } catch (error) {
    console.log('error', error);
    throw new Error('Failed to ingest your data');
  }
};

(async () => {
  await run();
  console.log('ingestion complete');
})();

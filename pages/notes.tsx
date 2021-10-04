import { NextPage } from 'next';
import Head from 'next/head';
import { getSortedPostsData } from '../lib/posts';
import PostList from '../components/PostList';

const Notes: NextPage = ({ posts }: any) => {
  return (
    <>  
      <Head>
        <title>{`Gurchet's Development Notes`}</title>
        <meta name="description" content="Software development notes" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <h1>Gurchet Rai - Notes</h1>
      <PostList posts={posts} /> 
    </>
  )
}

export async function getStaticProps() {
  const allPostsData = getSortedPostsData('notes');
  return {
    props: {
      posts: allPostsData
    }
  }
}

export const config = {
  unstable_runtimeJS: false
}

export default Notes

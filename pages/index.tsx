import { NextPage } from 'next';
import Head from 'next/head';
import { getSortedPostsData } from '../lib/posts';
import PostList from '../components/PostList';

const Home: NextPage = ({ posts }: any) => {
  return (
    <>  
      <Head>
        <title>{`Gurchet's Development Blog`}</title>
        <meta name="description" content="Software development blog" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <h1>Gurchet Rai</h1>
      <PostList posts={posts} /> 
    </>
  )
}

export async function getStaticProps() {
  const allPostsData = getSortedPostsData('post');
  return {
    props: {
      posts: allPostsData
    }
  }
}

export const config = {
  unstable_runtimeJS: false
}

export default Home

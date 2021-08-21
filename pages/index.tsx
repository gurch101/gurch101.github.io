import { NextPage } from 'next';
import Link from 'next/link';
import Head from 'next/head';
import Image from 'next/image';
import styles from '../styles/Home.module.css';
import { Post, getSortedPostsData } from '../lib/posts';
import PostMetadata from '../components/PostMetadata';

const PostListItem = ({ post }: any) => (
  <li className={styles.li}>
    <h2 className={styles.title}><Link href={`/${post.id}`}>{post.title}</Link></h2>
    <PostMetadata date={post.date} category={post.category} />
    <p className={styles.description}>{post.description}</p>
  </li>  
)

const Home: NextPage = ({ posts }: any) => {
  return (
    <>  
      <Head>
        <title>{`Gurchet's Development Blog`}</title>
        <meta name="description" content="Software development blog" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <h1>Gurchet Rai</h1>
      <ul className={styles.posts}>
        {posts.map((post : Post) => <PostListItem key={post.id} post={post} />)}
      </ul>      
    </>
  )
}

export async function getStaticProps() {
  const allPostsData = getSortedPostsData();
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

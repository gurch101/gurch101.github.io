import Head from 'next/head';
import { getAllPostIds, getPostData, PostIDRecord } from '../lib/posts';
import PostMetadata from '../components/PostMetadata';

export default function Post({ post }: any) {
    return (
        <div>
            <Head>
                <title>{`post.title | Gurchet's Development Blog`}</title>
                <meta name="description" content={post.description} />
                <link rel="icon" href="/favicon.ico" />
            </Head>            
            <h1 className="mb3">{post.title}</h1>
            <PostMetadata date={post.date} category={post.category} />
            <div className="mt30" dangerouslySetInnerHTML={{ __html: post.contentHtml }} />
        </div>
    )
}

export function getStaticPaths() {
    const paths = getAllPostIds();

    return {
        paths,
        fallback: false
    }
}

export async function getStaticProps({ params }: PostIDRecord ) {
    const postData = await getPostData(params.id)
    return {
      props: {
        post: postData
      }
    }
}

export const config = {
    unstable_runtimeJS: false
}
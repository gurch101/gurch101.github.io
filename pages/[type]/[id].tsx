import Head from "next/head";
import { PostMetadata } from "../../components/PostList";
import { getAllPosts, getPostData } from "../../lib/posts";

export default function Post({ post }: any) {
  return (
    <div>
      <Head>
        <title>{`${post.title} | Gurchet's Development Blog`}</title>
        <meta name="description" content={post.description} />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <h1 className="mb3">{post.title}</h1>
      <PostMetadata date={post.date} category={post.category} />
      <div
        className="mt25 post"
        dangerouslySetInnerHTML={{ __html: post.contentHtml }}
      />
    </div>
  );
}

export function getStaticPaths() {
  const paths = getAllPosts();
  return {
    paths,
    fallback: false,
  };
}

export async function getStaticProps(obj: any) {
  console.log(obj);
  const params = obj.params;
  const postData = await getPostData(params.type, params.id);
  return {
    props: {
      post: postData,
    },
  };
}

export const config = {
  unstable_runtimeJS: false,
};

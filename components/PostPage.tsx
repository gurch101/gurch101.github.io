import Head from "next/head";
import PostList from "./PostList";

interface SiteHeaderProps {
  title: string;
  description: string;
  header: string;
  posts: any;
}

export default function PostPage({
  title,
  description,
  header,
  posts,
}: SiteHeaderProps) {
  return (
    <>
      <Head>
        <title>{title}</title>
        <meta name="description" content={description} />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <h1>{header}</h1>
      <PostList posts={posts} />
    </>
  );
}

import { NextPage } from "next";
import PostPage from "../components/PostPage";
import { getSortedPostsData } from "../lib/posts";

const Notes: NextPage = ({ posts }: any) => {
  return (
    <PostPage
      title="Gurchet's Book Summaries"
      description="Book summaries"
      header="Gurchet Rai - Book Summaries"
      posts={posts}
    />
  );
};

export async function getStaticProps() {
  const allPostsData = getSortedPostsData("books");
  return {
    props: {
      posts: allPostsData,
    },
  };
}

export const config = {
  unstable_runtimeJS: false,
};

export default Notes;

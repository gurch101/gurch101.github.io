import { NextPage } from "next";
import PostPage from "../components/PostPage";
import { getSortedPostsData } from "../lib/posts";

const Notes: NextPage = ({ posts }: any) => {
  return (
    <PostPage
      title="Gurchet's Development Notes"
      description="Software development notes"
      header="Gurchet Rai - Notes"
      posts={posts}
    />
  );
};

export async function getStaticProps() {
  const allPostsData = getSortedPostsData("notes");
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

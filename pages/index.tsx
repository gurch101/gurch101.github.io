import { NextPage } from "next";
import PostPage from "../components/PostPage";
import { getSortedPostsData } from "../lib/posts";

const Home: NextPage = ({ posts }: any) => {
  return (
    <PostPage
      title="Gurchet's Development Blog"
      description="Software development blog"
      header="Gurchet Rai"
      posts={posts}
    />
  );
};

export async function getStaticProps() {
  const allPostsData = getSortedPostsData("blog");
  return {
    props: {
      posts: allPostsData,
    },
  };
}

export const config = {
  unstable_runtimeJS: false,
};

export default Home;

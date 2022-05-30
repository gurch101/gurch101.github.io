import * as fs from "fs";
import matter from "gray-matter";
import * as path from "path";
import { remark } from "remark";
import html from "remark-html";

export interface Post {
  id: string;
  title: string;
  date: Date;
  category: string;
  description: string;
  type: string;
}

export interface PostContent {
  contentHtml: string;
}

export interface PostMeta {
  id: string;
  type: PostType;
}

export interface PostRecord {
  params: PostMeta;
}

type PostType = "blog" | "books" | "notes";

const postsDirectory = (postType: PostType) =>
  path.join(process.cwd(), `posts/${postType}`);

const getParsedPostData = (postType: PostType, fileName: string): Post => {
  const id = fileName.replace(/\.md$/, "");
  const fullPath = path.join(postsDirectory(postType), fileName);
  const fileContents = fs.readFileSync(fullPath, "utf8");

  const matterResult = matter(fileContents);

  return {
    id,
    title: matterResult.data.title,
    date: matterResult.data.date,
    category: matterResult.data.category,
    description: matterResult.data.description,
    type: matterResult.data.type,
    ...matterResult.data,
  };
};

export function getSortedPostsData(postType: PostType) {
  const fileNames = fs.readdirSync(postsDirectory(postType));
  const allPostsData = fileNames
    .map((fileName) => getParsedPostData(postType, fileName))
    .filter((post) => post.type === postType);
  return allPostsData
    .sort(({ date: a }, { date: b }) => {
      if (a < b) {
        return 1;
      } else if (a > b) {
        return -1;
      } else {
        return 0;
      }
    })
    .map((post) => ({
      ...post,
      date: post.date.toDateString(),
    }));
}

function getPostsOfType(postType: PostType) {
  const fileNames = fs.readdirSync(postsDirectory(postType));
  return fileNames.map((fileName) => ({
    params: {
      id: fileName.replace(/\.md$/, ""),
      type: postType,
    },
  }));
}

export function getAllPosts(): PostRecord[] {
  return [
    ...getPostsOfType("blog"),
    ...getPostsOfType("books"),
    ...getPostsOfType("notes"),
  ];
}

export async function getPostData(
  type: PostType,
  id: string
): Promise<Post & PostContent> {
  const fullPath = path.join(postsDirectory(type), `${id}.md`);
  const fileContents = fs.readFileSync(fullPath, "utf8");

  const matterResult = matter(fileContents);

  const processedContent = await remark()
    .use(require("remark-prism"))
    .use(html)
    .process(matterResult.content);
  const contentHtml = processedContent.toString();

  return {
    id,
    title: matterResult.data.title,
    category: matterResult.data.category,
    description: matterResult.data.description,
    type: matterResult.data.type,
    ...matterResult.data,
    date: matterResult.data.date.toDateString(),
    contentHtml,
  };
}

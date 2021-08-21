import * as fs from 'fs'
import * as path from 'path'
import matter from 'gray-matter'
import { remark } from 'remark'
import html from 'remark-html'

export interface Post {
    id: string
    title: string
    date: Date
    category: string
    description: string
}

export interface PostContent {
    contentHtml: string
}

export interface PostID {
    id: string
}

export interface PostIDRecord {
    params: PostID
}

const postsDirectory = path.join(process.cwd(), 'posts');

const getParsedPostData = (fileName: string): Post => {
    const id = fileName.replace(/\.md$/, '')
    const fullPath = path.join(postsDirectory, fileName);
    const fileContents = fs.readFileSync(fullPath, 'utf8')

    const matterResult = matter(fileContents)

    return {
        id,
        title: matterResult.data.title,
        date: matterResult.data.date,
        category: matterResult.data.category,
        description: matterResult.data.description,
        ...matterResult.data
    }
}

export function getSortedPostsData() {
    const fileNames = fs.readdirSync(postsDirectory);
    const allPostsData = fileNames.map(getParsedPostData)
    return allPostsData.sort(({ date: a }, { date: b }) => {
        if (a < b) {
          return 1
        } else if (a > b) {
          return -1
        } else {
          return 0
        }
    }).map(post => ({
        ...post,
        date: post.date.toDateString()
    }));    
}

export function getAllPostIds(): PostIDRecord[] {
    const fileNames = fs.readdirSync(postsDirectory);
    return fileNames.map(fileName => ({
        params: {
            id: fileName.replace(/\.md$/, '')
        }
    }));   
}

export async function getPostData(id: string): Promise<Post & PostContent> {
    const fullPath = path.join(postsDirectory, `${id}.md`)
    const fileContents = fs.readFileSync(fullPath, 'utf8')
  
    const matterResult = matter(fileContents)
  
    const processedContent = await remark()
    .use(require('remark-prism'))
    .use(html)
    .process(matterResult.content)
  const contentHtml = processedContent.toString()

    return {
      id,
      title: matterResult.data.title,
      category: matterResult.data.category,
      description: matterResult.data.description,      
      ...matterResult.data,
      date: matterResult.data.date.toDateString(),
      contentHtml,
    }
}
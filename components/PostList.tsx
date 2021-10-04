import Link from 'next/link'
import styles from '../styles/PostList.module.css';
import { Post } from '../lib/posts';

interface PostMetadataProps {
    date: string
    category: string
}

const PostMetadata = ({ date, category }: PostMetadataProps) => (
    <div className='meta'>
        <span>{date}</span>
        <span>{category}</span>
    </div>
)

const PostListItem = ({ post }: any) => (
    <li className={styles.li}>
      <h2 className={styles.title}><Link href={`/${post.id}`}>{post.title}</Link></h2>
      <PostMetadata date={post.date} category={post.category} />
      <p className={styles.description}>{post.description}</p>
    </li>  
)

const PostList = ({ posts }: any) => (
    <ul className={styles.posts}>
        {posts.map((post : Post) => <PostListItem key={post.id} post={post} />)}
    </ul>         
)

export default PostList;

export { PostMetadata }
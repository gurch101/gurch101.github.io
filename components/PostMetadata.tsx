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

export default PostMetadata;
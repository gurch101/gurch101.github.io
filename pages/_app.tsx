import '../styles/globals.css'
import { AppProps } from 'next/app'
import "prismjs/themes/prism-tomorrow.css";

function MyApp({ Component, pageProps }: AppProps) {
  return (
    <div className="container">
      <Component {...pageProps} />
    </div>
  )
}

export const config = {
  unstable_runtimeJS: false
}

export default MyApp
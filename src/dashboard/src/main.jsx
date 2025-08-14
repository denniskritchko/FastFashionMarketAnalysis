import React, { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import axios from 'axios'

function App() {
	const [rows, setRows] = useState([])
	const [loading, setLoading] = useState(true)
	useEffect(() => {
		(async () => {
			try {
				const { data } = await axios.get('/api/brand-metrics')
				setRows(typeof data === 'string' ? JSON.parse(data) : data)
			} catch (e) {
				console.error(e)
			} finally {
				setLoading(false)
			}
		})()
	}, [])
	return (
		<div style={{ fontFamily: 'Inter, system-ui, Arial', padding: 24 }}>
			<h1>Fashion Sustainability & Sentiment</h1>
			{loading ? <p>Loading…</p> : (
				<table border="1" cellPadding="8">
					<thead>
						<tr>
							<th>Brand</th>
							<th>Avg Price</th>
							<th>Avg Sentiment</th>
						</tr>
					</thead>
					<tbody>
						{rows.map((r, i) => (
							<tr key={i}>
								<td>{r.brand}</td>
								<td>{r.avg_price?.toFixed ? r.avg_price.toFixed(2) : r.avg_price}</td>
								<td>{r.avg_sentiment?.toFixed ? r.avg_sentiment.toFixed(3) : r.avg_sentiment}</td>
							</tr>
						))}
					</tbody>
				</table>
			)}
		</div>
	)
}

createRoot(document.getElementById('root')).render(<App />)

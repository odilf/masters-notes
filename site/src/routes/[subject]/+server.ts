export async function GET({ params, fetch }) {
	const githubResponse = await fetch(
		`https://github.com/odilf/masters-notes/releases/latest/download/${params.subject}.pdf`
	);

	const newHeaders = new Headers(githubResponse.headers);
	newHeaders.set('Content-Type', 'application/pdf');
	newHeaders.set('Content-Disposition', `inline; filename="${params.subject}.pdf"`);

	return new Response(githubResponse.body, { headers: newHeaders });
}

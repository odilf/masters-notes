<script lang="ts">
	const { data } = $props();

	const url = $derived(
		`https://github.com/odilf/masters-notes/releases/latest/download/${data.subject}.pdf`
	);
</script>

<!-- <iframe
	src="https://mozilla.github.io/pdf.js/web/viewer.html?file={url}"
	style="width: 100%; height: 100vh;"
></iframe> -->

<embed src={url} type="application/pdf" width="100%" height="100%" />

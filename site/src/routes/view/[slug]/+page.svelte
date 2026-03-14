<script lang="ts">
	import { page } from "$app/state";

	const url = $derived(
		`https://github.com/odilf/masters-notes/releases/latest/download/${page.params.slug}.pdf`
	);
</script>

<object data={url} type="application/pdf" width="100%" height="100dvh" title={page.params.slug}></object>

<style>
    object {
        height: 100dvh;
    }
</style>

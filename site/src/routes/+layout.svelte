<script lang="ts">
	import '../app.css';
	import '@fontsource/iosevka/300';
	import '@fontsource/iosevka/400';
	import '@fontsource/iosevka/700';
	import '@fontsource/iosevka/900';
	import favicon from '$lib/assets/favicon.svg';

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

{@render children?.()}

<style>
	:global(body) {
		background-color: #222;
		color: #fff;
	}
</style>

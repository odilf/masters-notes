<script lang="ts">
	import { resolve } from '$app/paths';

	const { data } = $props();
</script>

<main class="mx-auto max-w-xl py-8">
	<h1 class="text-5xl">Notes for Master's in Computational and Applied Science</h1>
	<p class="mb-4 text-xl opacity-50">Carlos III University of Madrid</p>

	<h2 class="bold text-3xl">Subjects:</h2>
	<ul class="grid grid-cols-2">
		{#each data.subjects as subject (subject)}
			<li class="my-6">
				<h3 class="bold pb-2 text-2xl capitalize">{subject.replaceAll('-', ' ')}</h3>
				<a class="rounded bg-white p-2 text-black" href={resolve('/[subject]', { subject })}>
					Open in browser
				</a>
				<a
					class="rounded bg-white p-2 text-black opacity-90"
					href="https://github.com/odilf/masters-notes/releases/latest/download/{subject}.pdf"
				>
					Download
				</a>
			</li>
		{/each}
	</ul>
</main>

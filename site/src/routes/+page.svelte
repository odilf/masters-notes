<script lang="ts">
	import { Download, Globe } from '@lucide/svelte';
	import { resolve } from '$app/paths';

	const documents = [
  		{
    		"Analysis": {
     			Notes: "analysis",
     			Exercises: "analysis-exercises"
    		},
    		"Differential Equations": {
     			Notes: "differential-equations",
     			Exercises: "differential-equations-exercises"
    		},
    		"Modeling": {
     			Notes: "modeling"
    		},
    		"Linear Algebra": {
     			Notes: "linear-algebra"
    		}
  		},
		{
      		"Discrete Mathematics": {
     			Notes: "discrete"
      		},
      		"Optimization": {
     			Notes: "optimization"
      		},
      		"Stochastics": {
     			Notes: "stochastics"
      		}
		}

		] as Record<string, Record<string, string>>[];
	</script>

<main class="mx-auto max-w-3xl py-8 font-mono">
	<p class="text-2xl"> Notes for </p>
	<h1 class="text-5xl font-black mb-4 tracking-tight"> Master's in Computational and Applied Mathematics</h1>
	<p class="mb-8 text-xl opacity-50 font-light">At Carlos III University of Madrid</p>

	<p class="opacity-70"> The following are my personal notes I took at my Master's. These are provided with <b> no warrany of any kind </b> including but not limited to accuracy or correctness. Many notes are incomplete. </p>

	<ul class="grid grid-cols-2 gap-4">
		{#each documents as semester, i (i)}
		{#each Object.entries(semester) as [subject, docs] (subject)}
			<li class="my-6">
				<h3 class="font-bold text-2xl capitalize">{subject}</h3>
						<div class="list-disc grid-cols-[auto_auto_auto_1fr] grid items-center">
		    {#each Object.entries(docs) as [name, slug] (name)}
						<h4 class="font-light"> {name}: </h4>
				<a class="hover:bg-white/20 p-2 rounded transition transition-50" href={resolve('/view/[slug]', { slug })}>
					<Globe color="white" />
				</a>
				<a
					class="hover:bg-white/20 p-2 rounded transition transition-50 opacity-90"
					href="https://github.com/odilf/masters-notes/releases/latest/download/{slug}.pdf"
				>
					<Download color="white" />
				</a>
				<div></div>
		{/each}
						</div>
			</li>
		{/each}
		{/each}
	</ul>
</main>

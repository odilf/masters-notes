import { readdir } from 'node:fs/promises';

export const load = async () => {
	const subjects = (await readdir('../content')).filter((file) => !file.endsWith('.typ'));
	return {
		subjects
	};
};

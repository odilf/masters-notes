export const load = async ({ params }) => {
	const subject = params.subject;
	// const url = `https://github.com/odilf/masters-notes/releases/latest/download/${subject}.pdf`;
	return { subject };
};

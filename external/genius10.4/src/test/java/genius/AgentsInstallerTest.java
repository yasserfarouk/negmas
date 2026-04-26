package genius;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

public class AgentsInstallerTest {

	private final static File boapartyrepo = new File("boapartyrepo.xml");
	private final static File boarepo = new File("boarepository.xml");
	private final static File partyrepo = new File("partyrepository.xml");

	@Test
	public void testRun() throws IOException {
		clean();
		AgentsInstaller.run();
		assertTrue("boarepository was not installed properly", boarepo.exists());
		assertTrue("boapartyrepository was not installed properly", boapartyrepo.exists());
		assertTrue("partyrepository was not installed properly", partyrepo.exists());
		clean();
	}

	private void clean() {
		if (boarepo.exists()) {
			boarepo.delete();
		}
		if (boapartyrepo.exists()) {
			boapartyrepo.delete();
		}
		if (partyrepo.exists()) {
			partyrepo.delete();
		}
	}
}

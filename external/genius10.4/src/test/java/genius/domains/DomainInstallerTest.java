package genius.domains;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;

import org.junit.Test;

public class DomainInstallerTest {
	private final File domainrepo = new File("./domainrepository.xml");
	private final File etc = new File("etc");
	private final File partyprofile = new File("etc/templates/partydomain/party1_utility.xml");

	@Test
	public void test() throws IOException, URISyntaxException {
		if (domainrepo.exists()) {
			domainrepo.delete();
		}
		if (etc.exists()) {
			etc.delete();
		}

		DomainInstaller.run();
		DomainInstaller.copyRecursively("/genius/templates", "etc");
		assertTrue("domainrepo was not copied", domainrepo.exists());
		assertTrue("etc (templates folder) was not copied", etc.exists());
		assertTrue("party profile was not at expected location", partyprofile.exists());

		domainrepo.delete();
		deleteFolder(etc);
	}

	/**
	 * Clean up a folder (...)
	 * 
	 * @param folder
	 */
	private void deleteFolder(File folder) {
		File[] files = folder.listFiles();
		if (files != null) { // some JVMs return null for empty dirs
			for (File f : files) {
				if (f.isDirectory()) {
					deleteFolder(f);
				} else {
					f.delete();
				}
			}
		}
		folder.delete();
	}
}

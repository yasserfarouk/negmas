package genius;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

public class ProtocolsInstallerTest {

	private final static File protocolrepo = new File("protocolrepository.xml");

	private final static File multiprotocolrepo = new File("multipartyprotocolrepository.xml");

	@Test
	public void testRun() throws IOException {
		clean();
		ProtocolsInstaller.run();
		assertTrue("protocolrepository was not installed properly", protocolrepo.exists());
		assertTrue("partyrepository was not installed properly", multiprotocolrepo.exists());
		clean();
	}

	private void clean() {
		if (protocolrepo.exists()) {
			protocolrepo.delete();
		}
		if (multiprotocolrepo.exists()) {
			multiprotocolrepo.delete();
		}
	}
}

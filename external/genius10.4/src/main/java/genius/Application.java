package genius;

import java.io.IOException;

import genius.domains.DomainInstaller;
import genius.gui.MainPanel;

public class Application {
	public static void main(String[] args) throws IOException {
		ProtocolsInstaller.run();
		DomainInstaller.run();
		AgentsInstaller.run();

		MainPanel mainpanel = new MainPanel();
		mainpanel.pack();
		mainpanel.setVisible(true);
	}
}

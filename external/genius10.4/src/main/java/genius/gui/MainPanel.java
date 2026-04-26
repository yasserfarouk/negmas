package genius.gui;

import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Frame;
import java.awt.Toolkit;
import java.awt.event.MouseEvent;

import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;
import javax.swing.JTabbedPane;

import genius.gui.actions.AboutAction;
import genius.gui.actions.OpenManual;
import genius.gui.actions.Session;
import genius.gui.actions.Tournament;
import genius.gui.boaframework.BOARepositoryUI;
import genius.gui.boaparties.BoaPartiesPanel;
import genius.gui.domainrepository.DomainRepositoryUI;
import genius.gui.panels.tab.CloseTabbedPane;
import genius.gui.repository.PartyRepositoryUI;

/**
 * main application and main GUI panel.
 * 
 */
@SuppressWarnings("serial")
public class MainPanel extends JFrame implements GeniusAppInterface {

	private JTabbedPane repoArea = new JTabbedPane();
	private CloseTabbedPane editArea = new CloseTabbedPane();

	public MainPanel() {
		setLayout(new BorderLayout());
		
		// Fit nicely on smaller and bigger screens
		Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
		int prefx = 1280;
		int prefy = 1024;
		if (screenSize.getWidth() >= prefx && screenSize.getHeight() >= prefy)
			setPreferredSize(new Dimension(prefx, prefy));
		else
			setMinimumSize(new Dimension(600, 400));
		
		String version = getClass().getPackage().getImplementationVersion();
		if (version != null) // if the version is defined in the MANIFEST file,
								// e.g. Genius is in a .jar
			setTitle("GENIUS " + version);
		else
			setTitle("GENIUS");
		JSplitPane splitpane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT,
				repoArea, editArea);
		splitpane.setDividerLocation(300);

		add(splitpane, BorderLayout.CENTER);
		repoArea.addTab("Domains",
				new JScrollPane(new DomainRepositoryUI(this)));
		repoArea.addTab("BOA Components",
				new JScrollPane(new BOARepositoryUI(this)));
		repoArea.addTab("Parties", new PartyRepositoryUI());
		repoArea.addTab("Boa Parties", new BoaPartiesPanel());

		setJMenuBar(new MenuBar(this));

		editArea.addCloseListener((MouseEvent e, int overTabIndex) -> editArea
				.remove(overTabIndex));
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

	/**
	 * For testing
	 */
	public static void main(String[] args) {
		MainPanel mainpanel = new MainPanel();

		mainpanel.pack();
		mainpanel.setVisible(true);

	}

	@Override
	public void addTab(String title, Component comp) {
		editArea.addTab(title, comp);
		editArea.setSelectedComponent(comp);
	}

	@Override
	public Frame getMainFrame() {
		return this;
	}
}

@SuppressWarnings("serial")
class MenuBar extends JMenuBar {
	private JMenu startMenu = new JMenu();
	private JMenu helpMenu = new JMenu();

	public MenuBar(GeniusAppInterface mainPanel) {
		startMenu.setText("Start");
		startMenu.add(new Session(mainPanel));
		startMenu.add(new Tournament(mainPanel));

		add(startMenu);

		helpMenu.setText("Help");
		helpMenu.add(new OpenManual());
		helpMenu.add(new AboutAction());
		add(helpMenu);
	}
}
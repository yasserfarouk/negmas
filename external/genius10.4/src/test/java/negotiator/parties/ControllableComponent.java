package negotiator.parties;

import java.awt.Component;
import java.awt.Container;
import java.util.HashMap;

import javax.swing.JButton;

public class ControllableComponent {
	private HashMap<String, Component> componentMap = new HashMap<String, Component>();
	private Component topComponent;

	/**
	 * @param component
	 *            {@link Component}.
	 * @return map with all component names in a awt component.
	 */
	public ControllableComponent(Component component) {
		this.topComponent = component;
		findComponents(component);
	}

	/**
	 * Register given component and check all children of this component.
	 * 
	 * @param component
	 */
	private void findComponents(Component component) {
		componentMap.put(component.getName(), component);
		if (component instanceof Container) {
			for (Component child : ((Container) component).getComponents()) {
				findComponents(child);
			}
		}
	}

	/**
	 * Get a component with given name.
	 * 
	 * @param name
	 *            name of the component
	 * @return Component with given name, or null
	 */
	public Object get(String name) {
		return componentMap.get(name);
	}

	/**
	 * Click a button
	 * 
	 * @param name
	 *            the name of the button
	 * @throws if
	 *             there is no button with given name.
	 */
	public void clickButton(String name) {
		Component c = componentMap.get(name);
		if (c == null) {
			throw new IllegalArgumentException(
					"the component does not contain an element with name "
							+ name);
		}
		if (!(c instanceof JButton)) {
			throw new IllegalArgumentException("The component " + name
					+ " is not a JButton");
		}
		((JButton) c).doClick();
	}

	/**
	 * Waits till the component that we contain is actually visible.
	 * 
	 * @throws InterruptedException
	 */
	public void waitTillVisible() throws InterruptedException {
		while (!topComponent.isVisible()) {
			Thread.sleep(100);
		}
	}

	/**
	 * Wait in parallel thread till panel becomes visible. Then click on button.
	 * 
	 * @param control
	 * @param buttonName
	 */
	public void delayedClickOnButton(final String buttonName) {
		new Thread(new Runnable() {

			@Override
			public void run() {
				try {
					waitTillVisible();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				clickButton(buttonName);
			}
		}).start();

	}

}

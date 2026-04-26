/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package genius.gui.panels.tab;

import java.awt.event.MouseEvent;
import java.util.EventListener;


public interface CloseListener extends EventListener {
	public void closeOperation(MouseEvent e, int overTabIndex);
}

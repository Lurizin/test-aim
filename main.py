from icecream import ic
ic("Starting...")

import torch, mss
from ultralytics import YOLO
import win32api, win32con, win32gui
import numpy as np
from time import sleep
import cv2
import threading
import ctypes
from ctypes import wintypes

# ========== CONFIG ==========

MODEL = 'models/Arsenal (4K) by Themida.onnx'

BOX_SIZE = 200
# ↑BOX_SIZE -> ↓SPEED ↑DETECTION_AREA

STOP_KEY = win32con.VK_F1
STOPPED = False  # Iniciar com o sistema ativo

# Configurações anti-bloqueio
FORCE_CURSOR_CONTROL = False  # Forçar controle do cursor
USE_AGGRESSIVE_MODE = False   # Usar modo agressivo para contornar bloqueios

# ========== AIMBOT CONFIG ==========

SMOOTH_FACTOR = 0.15  # 0.1 = muito suave, 1.0 = instantâneo
AIM_KEY = win32con.VK_RBUTTON  # Botão direito para ativar aim
AUTO_SHOOT = False  # Se deve atirar automaticamente
HEADSHOT_OFFSET = -10  # Offset para mirar na cabeça (pixels acima do centro)
MIN_CONFIDENCE = 0.4  # Confiança mínima para aimbot (reduzida para melhor detecção)

# ========== VISUALIZAÇÃO CONFIG ==========

SHOW_DETECTIONS = True  # Mostrar retângulos das detecções
VISUAL_KEY = win32con.VK_F2  # F2 para toggle da visualização
detection_frame = None  # Frame global para visualização

# ===================================

# ============================

CUDA = torch.cuda.is_available()

ic(f"CUDA status: {CUDA}")

device = torch.device("cuda" if CUDA else "cpu")
model = YOLO(MODEL, task='detect')

sct = mss.mss()
screen_width, screen_height = sct.monitors[1]['width'], sct.monitors[1]['height']
crosshair_x, crosshair_y = screen_width // 2, screen_height // 2

ALPHA = round(
    screen_width * screen_height * BOX_SIZE / (1920*1080)
)

CONFIDENCE = 0.5

ic(f"ALPHA value: {ALPHA}")

# Funções para controle avançado do cursor
def force_cursor_control():
    """Força o controle do cursor usando ClipCursor e outras técnicas"""
    if not FORCE_CURSOR_CONTROL:
        return
    
    try:
        # Obter handle da janela ativa
        hwnd = win32gui.GetForegroundWindow()
        
        # Obter retângulo da janela
        rect = win32gui.GetWindowRect(hwnd)
        
        # Usar ClipCursor para confinar o cursor à janela
        ctypes.windll.user32.ClipCursor(ctypes.byref(wintypes.RECT(*rect)))
        
        ic(f"Cursor confinado à janela: {rect}")
    except Exception as e:
        ic(f"Erro ao confinar cursor: {e}")

def release_cursor_control():
    """Libera o controle forçado do cursor"""
    try:
        # Liberar ClipCursor
        ctypes.windll.user32.ClipCursor(None)
        ic("Cursor liberado")
    except Exception as e:
        ic(f"Erro ao liberar cursor: {e}")

def aggressive_mouse_move(x, y):
    """Movimento de mouse agressivo usando múltiplas APIs"""
    if not USE_AGGRESSIVE_MODE:
        return False
    
    success = False
    
    # Método 1: SetCursorPos direto
    try:
        result = ctypes.windll.user32.SetCursorPos(int(x), int(y))
        if result:
            success = True
    except:
        pass
    
    # Método 2: SendInput com movimento absoluto
    try:
        # Converter para coordenadas normalizadas (0-65535)
        screen_width = win32api.GetSystemMetrics(0)
        screen_height = win32api.GetSystemMetrics(1)
        
        norm_x = int((x * 65535) / screen_width)
        norm_y = int((y * 65535) / screen_height)
        
        # Estruturas para SendInput
        PUL = ctypes.POINTER(ctypes.c_ulong)
        
        class MouseInput(ctypes.Structure):
            _fields_ = [("dx", ctypes.c_long),
                       ("dy", ctypes.c_long),
                       ("mouseData", ctypes.c_ulong),
                       ("dwFlags", ctypes.c_ulong),
                       ("time", ctypes.c_ulong),
                       ("dwExtraInfo", PUL)]
        
        class Input_I(ctypes.Union):
            _fields_ = [("mi", MouseInput)]
        
        class Input(ctypes.Structure):
            _fields_ = [("type", ctypes.c_ulong),
                       ("ii", Input_I)]
        
        # Movimento absoluto
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.mi = MouseInput(norm_x, norm_y, 0, 0x8001, 0, ctypes.pointer(extra))  # MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE
        input_struct = Input(ctypes.c_ulong(0), ii_)
        
        result = ctypes.windll.user32.SendInput(1, ctypes.pointer(input_struct), ctypes.sizeof(input_struct))
        if result:
            success = True
    except:
        pass
    
    # Método 3: Múltiplas chamadas rápidas
    if not success:
        try:
            for _ in range(3):
                ctypes.windll.user32.SetCursorPos(int(x), int(y))
                sleep(0.001)
            success = True
        except:
            pass
    
    return success

def capture_screen():
    monitor = {
        "top": crosshair_y - ALPHA,
        "left": crosshair_x - ALPHA,
        "width": 2*ALPHA,
        "height": 2*ALPHA
    }

    screenshot = sct.grab(monitor)
    return np.array(screenshot)[:, :, :3] # Remove alpha

def check_stop():
    key_state = win32api.GetAsyncKeyState(STOP_KEY)
    global STOPPED
    pressed = bool(key_state & 0x8000)
    if pressed == True:
        STOPPED = not STOPPED
        ic(f"{STOPPED=}")
        sleep(0.3)

def smooth_move_mouse(target_x, target_y):
    """Move o mouse suavemente para a posição alvo com movimento gradual"""
    current_x, current_y = win32api.GetCursorPos()
    
    # Calcular posição alvo absoluta na tela
    # target_x e target_y são coordenadas relativas à área de captura
    # Precisamos converter para coordenadas absolutas da tela
    capture_start_x = crosshair_x - ALPHA
    capture_start_y = crosshair_y - ALPHA
    
    target_screen_x = capture_start_x + target_x
    target_screen_y = capture_start_y + target_y + HEADSHOT_OFFSET
    
    ic(f"Posição atual: ({current_x}, {current_y}) -> Alvo: ({target_screen_x:.1f}, {target_screen_y:.1f})")
    
    # Calcular diferença total
    total_diff_x = target_screen_x - current_x
    total_diff_y = target_screen_y - current_y
    
    # Calcular distância total para determinar número de passos
    total_distance = (total_diff_x ** 2 + total_diff_y ** 2) ** 0.5
    
    # Se a distância for muito pequena, mover diretamente
    if total_distance < 5:
        try:
            win32api.SetCursorPos((int(target_screen_x), int(target_screen_y)))
            ic(f"Movimento direto para posição próxima")
        except Exception as e:
            ic(f"Movimento direto falhou: {e}")
        return
    
    # Movimento suave em múltiplos passos
    steps = max(5, int(total_distance / 10))  # Mais passos para distâncias maiores
    steps = min(steps, 20)  # Limitar máximo de passos
    
    ic(f"Movimento suave em {steps} passos, distância: {total_distance:.1f}px")
    
    for i in range(1, steps + 1):
        # Interpolação suave usando função ease-out
        progress = i / steps
        # Função ease-out para movimento mais natural
        eased_progress = 1 - (1 - progress) ** 2
        
        # Calcular posição intermediária
        intermediate_x = current_x + (total_diff_x * eased_progress)
        intermediate_y = current_y + (total_diff_y * eased_progress)
        
        try:
            # Tentar múltiplas técnicas de movimento
            success = False
            
            # Técnica 1: SetCursorPos padrão
            try:
                win32api.SetCursorPos((int(intermediate_x), int(intermediate_y)))
                success = True
            except:
                pass
            
            # Técnica 2: ctypes direto se a primeira falhar
            if not success:
                try:
                    ctypes.windll.user32.SetCursorPos(int(intermediate_x), int(intermediate_y))
                    success = True
                except:
                    pass
            
            # Técnica 3: Movimento relativo se as outras falharem
            if not success and i == 1:  # Só tentar na primeira iteração
                try:
                    step_x = total_diff_x * eased_progress
                    step_y = total_diff_y * eased_progress
                    dx = int(step_x * 65536 / win32api.GetSystemMetrics(0))
                    dy = int(step_y * 65536 / win32api.GetSystemMetrics(1))
                    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0)
                    success = True
                except:
                    pass
            
            # Delay entre movimentos para suavidade
            if i < steps:  # Não fazer delay no último movimento
                sleep(0.008)  # 8ms entre movimentos para suavidade
                
        except Exception as e:
            ic(f"Erro no passo {i}: {e}")
            continue
    
    ic(f"Movimento suave concluído")

def click():
    """Simula um clique do mouse com múltiplas técnicas"""
    # Técnica 1: mouse_event padrão
    try:
        x, y = win32api.GetCursorPos()
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
        sleep(0.06)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)
    except:
        pass
    
    # Técnica 2: SendInput para contornar bloqueios
    try:
        # Estrutura INPUT para SendInput
        PUL = ctypes.POINTER(ctypes.c_ulong)
        
        class KeyBdInput(ctypes.Structure):
            _fields_ = [("wVk", ctypes.c_ushort),
                       ("wScan", ctypes.c_ushort),
                       ("dwFlags", ctypes.c_ulong),
                       ("time", ctypes.c_ulong),
                       ("dwExtraInfo", PUL)]
        
        class HardwareInput(ctypes.Structure):
            _fields_ = [("uMsg", ctypes.c_ulong),
                       ("wParamL", ctypes.c_short),
                       ("wParamH", ctypes.c_ushort)]
        
        class MouseInput(ctypes.Structure):
            _fields_ = [("dx", ctypes.c_long),
                       ("dy", ctypes.c_long),
                       ("mouseData", ctypes.c_ulong),
                       ("dwFlags", ctypes.c_ulong),
                       ("time", ctypes.c_ulong),
                       ("dwExtraInfo", PUL)]
        
        class Input_I(ctypes.Union):
            _fields_ = [("ki", KeyBdInput),
                       ("mi", MouseInput),
                       ("hi", HardwareInput)]
        
        class Input(ctypes.Structure):
            _fields_ = [("type", ctypes.c_ulong),
                       ("ii", Input_I)]
        
        # Clique para baixo
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.mi = MouseInput(0, 0, 0, 0x0002, 0, ctypes.pointer(extra))  # MOUSEEVENTF_LEFTDOWN
        x = Input(ctypes.c_ulong(0), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        
        sleep(0.01)
        
        # Clique para cima
        ii_.mi = MouseInput(0, 0, 0, 0x0004, 0, ctypes.pointer(extra))  # MOUSEEVENTF_LEFTUP
        x = Input(ctypes.c_ulong(0), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    except:
        pass

def draw_detections(frame, results):
    """Desenha retângulos e informações das detecções no frame"""
    global detection_frame
    detection_frame = frame.copy()
    
    # Contador de detecções
    total_detections = 0
    valid_targets = 0
    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:  # Class id for person
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                total_detections += 1
                
                # Cor baseada na confiança
                if conf >= MIN_CONFIDENCE:
                    color = (0, 255, 0)  # Verde para alvos válidos
                    thickness = 3
                    valid_targets += 1
                    # Desenhar círculo maior para alvos válidos
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    cv2.circle(detection_frame, (center_x, center_y), 15, color, 3)
                else:
                    color = (0, 165, 255)  # Laranja para baixa confiança
                    thickness = 2
                
                # Desenhar retângulo principal
                cv2.rectangle(detection_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Desenhar centro do alvo
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(detection_frame, (center_x, center_y), 5, color, -1)
                
                # Desenhar linha do centro da tela até o alvo
                screen_center_x, screen_center_y = ALPHA, ALPHA
                cv2.line(detection_frame, (screen_center_x, screen_center_y), 
                        (center_x, center_y), (255, 255, 0), 1)
                
                # Calcular distância
                distance = ((center_x - screen_center_x) ** 2 + (center_y - screen_center_y) ** 2) ** 0.5
                
                # Texto com confiança e distância
                label = f"Person {conf:.2f} | Dist: {distance:.0f}px"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Fundo do texto
                cv2.rectangle(detection_frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0] + 5, y1), color, -1)
                cv2.putText(detection_frame, label, (x1 + 2, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Desenhar crosshair no centro (mais visível)
    center_x, center_y = ALPHA, ALPHA
    # Crosshair principal
    cv2.line(detection_frame, (center_x - 30, center_y), (center_x + 30, center_y), (0, 0, 255), 3)
    cv2.line(detection_frame, (center_x, center_y - 30), (center_x, center_y + 30), (0, 0, 255), 3)
    # Círculo central
    cv2.circle(detection_frame, (center_x, center_y), 10, (0, 0, 255), 2)
    
    # Área de captura (retângulo da região analisada)
    cv2.rectangle(detection_frame, (0, 0), (ALPHA*2, ALPHA*2), (255, 0, 255), 2)
    
    # Informações no canto superior esquerdo
    info_y = 30
    cv2.putText(detection_frame, f"Deteccoes: {total_detections}", (10, info_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    info_y += 30
    cv2.putText(detection_frame, f"Alvos Validos: {valid_targets}", (10, info_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    info_y += 30
    cv2.putText(detection_frame, f"Min Conf: {MIN_CONFIDENCE}", (10, info_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return detection_frame

def show_detection_window():
    """Thread para mostrar janela de detecções"""
    global detection_frame, SHOW_DETECTIONS
    
    # Configurar janela para ser redimensionável
    cv2.namedWindow("AimmyV2 - Deteccoes YOLO", cv2.WINDOW_NORMAL)
    # Definir tamanho inicial maior (80% da tela)
    initial_width = int(screen_width * 0.8)
    initial_height = int(screen_height * 0.8)
    cv2.resizeWindow("AimmyV2 - Deteccoes YOLO", initial_width, initial_height)
    
    while True:
        if SHOW_DETECTIONS and detection_frame is not None:
            # Redimensionar o frame para um tamanho maior se necessário
            display_frame = detection_frame.copy()
            
            # Se o frame for muito pequeno, redimensionar para um tamanho mínimo
            min_size = 800
            if display_frame.shape[0] < min_size or display_frame.shape[1] < min_size:
                scale_factor = min_size / min(display_frame.shape[0], display_frame.shape[1])
                new_width = int(display_frame.shape[1] * scale_factor)
                new_height = int(display_frame.shape[0] * scale_factor)
                display_frame = cv2.resize(display_frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Adicionar overlay com informações do sistema
            overlay = display_frame.copy()
            
            # Painel de informações no canto superior direito
            panel_width = 300
            panel_height = 150
            panel_x = display_frame.shape[1] - panel_width - 10
            panel_y = 10
            
            # Fundo semi-transparente para o painel
            cv2.rectangle(overlay, (panel_x, panel_y), 
                         (panel_x + panel_width, panel_y + panel_height), 
                         (0, 0, 0), -1)
            
            # Misturar overlay com transparência
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
            
            # Informações do painel
            info_x = panel_x + 10
            info_y = panel_y + 25
            line_height = 20
            
            cv2.putText(display_frame, "AimmyV2 - YOLO Vision", (info_x, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            info_y += line_height + 5
            
            cv2.putText(display_frame, "Controles:", (info_x, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            info_y += line_height
            
            cv2.putText(display_frame, "F1: Pausar/Iniciar", (info_x, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            info_y += line_height
            
            cv2.putText(display_frame, "F2: Toggle Visualizacao", (info_x, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            info_y += line_height
            
            cv2.putText(display_frame, "Botao Direito: Aimbot", (info_x, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            info_y += line_height
            
            cv2.putText(display_frame, "Q: Fechar Janela", (info_x, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Legenda de cores no canto inferior esquerdo
            legend_y = display_frame.shape[0] - 80
            cv2.putText(display_frame, "Legenda:", (10, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            legend_y += 25
            
            # Verde - Alvos válidos
            cv2.rectangle(display_frame, (10, legend_y - 15), (30, legend_y - 5), (0, 255, 0), -1)
            cv2.putText(display_frame, "Alvo Valido (Conf >= 0.4)", (35, legend_y - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            legend_y += 20
            
            # Laranja - Baixa confiança
            cv2.rectangle(display_frame, (10, legend_y - 15), (30, legend_y - 5), (0, 165, 255), -1)
            cv2.putText(display_frame, "Baixa Confianca (< 0.4)", (35, legend_y - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            legend_y += 20
            
            # Vermelho - Crosshair
            cv2.rectangle(display_frame, (10, legend_y - 15), (30, legend_y - 5), (0, 0, 255), -1)
            cv2.putText(display_frame, "Centro da Mira", (35, legend_y - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("AimmyV2 - Deteccoes YOLO", display_frame)
            
            # Verificar se janela foi fechada
            if cv2.waitKey(1) & 0xFF == ord('q'):
                SHOW_DETECTIONS = False
                break
        else:
            cv2.waitKey(100)  # Aguardar quando não há frame
    
    cv2.destroyAllWindows()

def check_visual_toggle():
    """Verifica se F2 foi pressionado para toggle da visualização"""
    global SHOW_DETECTIONS
    key_state = win32api.GetAsyncKeyState(VISUAL_KEY)
    pressed = bool(key_state & 0x8000)
    if pressed:
        SHOW_DETECTIONS = not SHOW_DETECTIONS
        status = "ATIVADA" if SHOW_DETECTIONS else "DESATIVADA"
        ic(f"Visualização {status}")
        
        if SHOW_DETECTIONS:
            # Criar janela se não existir
            cv2.namedWindow("AimmyV2 - Deteccoes", cv2.WINDOW_NORMAL)
        else:
            # Fechar janela
            cv2.destroyAllWindows()
        
        sleep(0.3)  # Prevenir múltiplos toggles

def aimbot():
    """Aimbot ativo que move o mouse para o alvo mais próximo"""
    try:
        frame = capture_screen()
        results = model(frame, verbose=False)
        
        # Desenhar detecções se visualização estiver ativa
        if SHOW_DETECTIONS:
            draw_detections(frame, results)
        
        closest_target = None
        min_distance = float('inf')
        targets_found = 0
        total_detections = 0
        
        # O centro da área capturada nas coordenadas locais
        capture_center_x = ALPHA
        capture_center_y = ALPHA
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:  # Class id for person
                    conf = float(box.conf[0])
                    total_detections += 1
                    
                    # Log todas as detecções para debug
                    ic(f"Detecção encontrada - Confiança: {conf:.3f}, Min requerida: {MIN_CONFIDENCE}")
                    
                    if conf < MIN_CONFIDENCE: 
                        ic(f"Detecção rejeitada por baixa confiança: {conf:.3f} < {MIN_CONFIDENCE}")
                        continue
                    
                    targets_found += 1
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Calcular centro do alvo nas coordenadas da área capturada
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    ic(f"Alvo detectado em: ({center_x:.1f}, {center_y:.1f}) na área capturada")
                    
                    # Calcular distância do centro da área capturada
                    distance = ((center_x - capture_center_x) ** 2 + (center_y - capture_center_y) ** 2) ** 0.5
                    
                    # Encontrar alvo mais próximo
                    if distance < min_distance:
                        min_distance = distance
                        closest_target = (center_x, center_y)
        
        # Debug: mostrar informações dos alvos
        if total_detections > 0:
            ic(f"Total detecções: {total_detections}, Alvos válidos: {targets_found}")
            if targets_found > 0:
                ic(f"Distância mínima: {min_distance:.1f}")
        else:
            ic("Nenhuma detecção encontrada na área de captura")
        
        # Mover para o alvo mais próximo
        if closest_target:
            ic(f"Movendo para alvo: ({closest_target[0]:.1f}, {closest_target[1]:.1f})")
            smooth_move_mouse(closest_target[0], closest_target[1])
            
            # Auto-shoot se habilitado e alvo está próximo do centro
            if AUTO_SHOOT and min_distance < 30:  # 30 pixels de tolerância
                ic("Auto-shoot ativado!")
                click()
                sleep(0.07)
    
    except Exception as e:
        ic(f"Erro no aimbot: {e}")

def triggerbot():
    """Triggerbot original (mantido para compatibilidade)"""
    frame = capture_screen()
    results = model(frame, verbose=False)
    
    # Desenhar detecções se visualização estiver ativa
    if SHOW_DETECTIONS:
        draw_detections(frame, results)
    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0: # Class id for person
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                if x1 <= ALPHA <= x2 and y1 <= ALPHA <= y2:
                    conf = float(box.conf[0])
                    if conf < CONFIDENCE: continue
                    click()
                    sleep(0.07)
                    return
                
def main():
    global STOPPED
    aimbot_active = False
    
    ic(f"{STOPPED=}")
    ic("The game should be on fullscreen mode!")
    ic(f"Press right mouse button to toggle aimbot")
    ic("Press F1 to toggle stop/start")
    ic("Press F2 to toggle detection visualization")
    
    # Iniciar thread de visualização
    if SHOW_DETECTIONS:
        visual_thread = threading.Thread(target=show_detection_window, daemon=True)
        visual_thread.start()
        ic("Visualização de detecções iniciada!")
    
    while True:
        try:
            check_stop()
            check_visual_toggle()  # Verificar toggle da visualização
            
            # Check if aim key is pressed to toggle aimbot
            aim_key_state = win32api.GetAsyncKeyState(AIM_KEY)
            if bool(aim_key_state & 0x8000):
                aimbot_active = not aimbot_active
                status = "ACTIVATED" if aimbot_active else "DEACTIVATED"
                ic(f"Aimbot {status}")
                sleep(0.3)  # Prevent multiple toggles
            
            if not STOPPED:
                if aimbot_active:
                    aimbot()  # Use active aimbot
                else:
                    triggerbot()  # Use original triggerbot

            sleep(0.01)
        except KeyboardInterrupt:
            ic("Quitting!")
            cv2.destroyAllWindows()  # Fechar janelas OpenCV
            break
        except Exception as e:
            ic(e)
            pass

if __name__ == '__main__':
    main()
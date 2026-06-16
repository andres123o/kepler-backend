. LO QUE REVELARON LOS DATOS — ANÁLISIS SQL
Se corrieron tres queries sobre trii-bi.scheduled_queries.user_attributes con datos desde 2022, Colombia, excluyendo bloqueados y revisar.
Hallazgo 1 — Efecto quincena: confirmado con 41% de diferencia
PeríodoDíasPct depósitosPico fin de mes28, 293.72%, 3.89%Pico inicio de mes1, 2, 33.54-3.67%Pico mitad de mes163.58%Valle profundo11, 12, 132.76-2.99%
El pico no es el día 15 o 30 exacto (día de pago) — es 1-2 días después. La gente recibe la quincena y actúa al día siguiente. Diferencia entre día pico (3.89%) y día valle (2.76%) = 41% más depósitos solo por timing de liquidez.
Hallazgo 2 — Ciclo del usuario: más concentrado de lo esperado
VentanaPct acumuladoDía 0 (mismo día aprobación)33.35%Días 0-148.16%Días 0-254.78%Días 0-7~72%Días 8-60~28%
La mitad de todos los depósitos ocurren en las primeras 48 horas post-aprobación. El modelo del sistema asumía una distribución hasta semana 5. La realidad es que la ventana crítica es día 0-2, con una cola de rescate hasta día 21.
Hay repuntes en días 14 y 21 — usuarios que coinciden con la siguiente quincena después de su aprobación.
Implicación estructural para C6:
Día 0: Push inmediato (ya existe)
Día 1: Email de urgencia máxima ← reforzar
Día 2: Push de refuerzo ← CRÍTICO
Día 7: Reactivación con señal de mercado
Día 14: Segunda reactivación (coincide con quincena)
Día 21: Último intento antes de probabilidad mínima
Hallazgo 3 — Efecto día de semana
DíaPct depósitosMartes18.68% — PICOMiércoles18.04%Jueves17.66%Viernes17.35%Lunes15.22%Sábado7.64%Domingo5.41%
Martes-Viernes concentran el 72% de todos los depósitos. El lunes baja porque es el día de intención — la acción ocurre el martes. Push enviado el lunes a las 8-9am genera el depósito del martes.
